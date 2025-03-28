# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import os
import time
import warnings
from typing import Callable, List, Optional, Union
from tqdm import tqdm

import datasets
import torch
from accelerate import Accelerator
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from ..core import (
    WANDB_PADDING,
    PPODecorators,
    clip_by_value,
    convert_to_scalar,
    entropy_from_logits,
    flatten_dict,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    masked_whiten,
    set_seed,
    stack_dicts,
    stats_to_np,
)
from ..import_utils import is_torch_greater_2_0
from ..models import (
    SUPPORTED_ARCHITECTURES,
    PreTrainedModelWrapper,
    create_reference_model,
)
from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig

from superhf.utils import print_memory_utilization


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
tags:
- trl
- transformers
- reinforcement-learning
---

# {model_name}

This is a [TRL language model](https://github.com/lvwerra/trl) that has been fine-tuned with reinforcement learning to
 guide the model outputs according to a value, function, or human feedback. The model can be used for text generation.

## Usage

To use this model for inference, first install the TRL library:

```bash
python -m pip install trl
```

You can then generate text as follows:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{model_id}")
outputs = generator("Hello, my llama is cute")
```

If you want to use the model for training or to obtain the outputs from the value head, load the model as follows:

```python
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLMWithValueHead.from_pretrained("{model_id}")

inputs = tokenizer("Hello, my llama is cute", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"])
```
"""


class PPOTrainer(BaseTrainer):
    """
    The PPOTrainer uses Proximal Policy Optimization to optimise language models.
    Note, this trainer is heavily inspired by the original OpenAI learning to summarize work here:
    https://github.com/openai/summarize-from-feedback

    Attributes:
        **config** (`PPOConfig`) -- Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more
         details.
        **model** (`PreTrainedModelWrapper`) -- Model to be optimized, Hugging Face transformer model with a value head.
            Check the documentation of `PreTrainedModelWrapper` for more details.
        **ref_model** (`PreTrainedModelWrapper`, *optional*) -- Reference model to be used for KL penalty, Hugging Face
            transformer model with a casual language modelling head. Check the documentation of `PreTrainedModelWrapper`
            for more details. If no reference model is provided, the trainer will create a reference model with the same
             architecture as the model to be optimized with shared layers.
        **tokenizer** (`Union[PreTrainedTokenizer, PreTrainedTokenizerFast]`) -- Tokenizer to be used for encoding the
            data. Check the documentation of `transformers.PreTrainedTokenizer` and
            `transformers.PreTrainedTokenizerFast` for more details.
        **dataset** (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*) -- PyTorch dataset or Hugging
            Face dataset. This is used to create a PyTorch dataloader. If no dataset is provided, the dataloader must be
             created outside the trainer users needs to design their own dataloader and make sure the batch
            size that is used is the same as the one specified in the configuration object.
        **optimizer** (`torch.optim.Optimizer`, *optional*) -- Optimizer to be used for training. If no optimizer is
            provided, the trainer will create an Adam optimizer with the learning rate specified in the configuration
            object.
        **data_collator** (DataCollatorForLanguageModeling, *optional*) -- Data collator to be used for training and
            passed along the dataloader
        **num_shared_layers** (int, *optional*) -- Number of layers to be shared between the model and the reference
            model, if no reference model is passed. If no number is provided, all the layers will be shared.
        **lr_scheduler** (`torch.optim.lr_scheduler`, *optional*) -- Learning rate scheduler to be used for training.
    """

    def __init__(
        self,
        config: PPOConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: PreTrainedModelWrapper = None,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator=None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        """
        Initialize PPOTrainer.

        Args:
            config (`PPOConfig`):
                Configuration object for PPOTrainer. Check the documentation of `PPOConfig` for more details.
            model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a value head.
            ref_model (`PreTrainedModelWrapper`):
                Hugging Face transformer model with a casual language modelling head. Used for KL penalty
            tokenizer (`transformers.PreTrainedTokenizer`):
                Hugging Face tokenizer
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`], *optional*):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model. If none is passed,
                a warning will be raised in a multi-GPU setting.
            optimizer (Optional[`torch.optim.Optimizer`]):
                Optimizer used for training. If `None`, the `Adam` is used as default.
            data_collator (Optional[function]):
                Data collator function.
            num_shared_layers (Optional[int]):
                Number of shared layers between the model and the reference model. If `None`, all layers are shared.
                used only if `ref_model` is `None`.
            lr_scheduler (Optional[`torch.optim.lr_scheduler`]):
                Learning rate scheduler used for training.
        """
        super().__init__(config)

        # initial seed for reproducible experiments
        set_seed(config.seed)

        # Step 0: check positional arguments validity
        if not isinstance(config, PPOConfig):
            raise ValueError(f"config must be a PPOConfig, got {type(config)}")
        if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                "tokenizer must be a PreTrainedTokenizer or PreTrainedTokenizerFast,"
                f" got {type(tokenizer)}"
            )
        if not isinstance(model, (SUPPORTED_ARCHITECTURES)):
            raise ValueError(
                f"model must be a PreTrainedModelWrapper, got {type(model)} - supported"
                f" architectures are: {SUPPORTED_ARCHITECTURES}"
            )
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            **config.accelerator_kwargs,
        )
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=dict(trl_ppo_trainer_config=config.to_dict()),
            init_kwargs=config.tracker_kwargs,
        )

        self.model = model
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)

        if isinstance(ref_model, SUPPORTED_ARCHITECTURES):
            self.ref_model = ref_model
            if num_shared_layers is not None:
                warnings.warn(
                    (
                        "num_shared_layers is ignored when ref_model is provided. Two"
                        " different models are used for the model and the reference"
                        " model and no layers are shared."
                    ),
                    UserWarning,
                )
        elif ref_model is None and not self.is_peft_model:
            self.ref_model = create_reference_model(
                self.model, num_shared_layers=num_shared_layers
            )
        elif self.is_peft_model:
            self.ref_model = None
        else:
            raise ValueError(
                "ref_model must be a PreTrainedModelWrapper or `None`, got"
                f" {type(ref_model)} - supported architectures are:"
                f" {SUPPORTED_ARCHITECTURES} "
            )

        if not (
            isinstance(tokenizer, PreTrainedTokenizer)
            or isinstance(tokenizer, PreTrainedTokenizerFast)
        ):
            raise ValueError(
                "tokenizer must be a transformers.PreTrainedTokenizer or"
                " transformers.PreTrainedTokenizerFast"
            )
        self.tokenizer = tokenizer

        if dataset is not None and not (
            isinstance(dataset, torch.utils.data.Dataset)
            or isinstance(dataset, Dataset)
        ):
            raise ValueError(
                "dataloader must be a torch.utils.data.Dataset or datasets.Dataset"
            )
        elif dataset is None:
            warnings.warn(
                (
                    "No dataset is provided. Make sure to set config.batch_size to the"
                    " correct value before training."
                ),
                UserWarning,
            )
        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        elif self.dataset is None and self.accelerator.num_processes > 1:
            # pylint: disable=too-many-function-args
            warnings.warn(
                (
                    "No dataset is provided. In a multi-GPU setting, this will lead to"
                    " an error. You should"
                ),
                (
                    " prepare your dataloader yourself with `dataloader ="
                    " ppo_trainer.accelerator.prepare(dataloader)`"
                ),
                (
                    " and using `torch.utils.data.DataLoader`, or pass a dataset to the"
                    " `PPOTrainer`. Please "
                ),
                " refer to the documentation for more details.",
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        # Step 3: Initialize optimizer and data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            lr_scheduler_class = (
                torch.optim.lr_scheduler._LRScheduler
                if not is_torch_greater_2_0()
                else torch.optim.lr_scheduler.LRScheduler
            )

            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    "lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or"
                    " torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)"
                )

        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(
                self.config.init_kl_coef, self.config.target, self.config.horizon
            )
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        # Safety checkers for DS integration
        is_deepspeed_used = (
            self.accelerator.distributed_type == "DEEPSPEED"
            and hasattr(self.accelerator.state, "deepspeed_plugin")
        )

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )
        if is_deepspeed_used:
            # 8 bit models are already set on the correct device
            if not self.is_peft_model and not getattr(
                self.ref_model.pretrained_model, "is_loaded_in_8bit", False
            ):
                # DS integration only allows for single model and as `ref_model` is only used for
                # `KL devergence loss`,i.e, in eval model, just have it be on the respective device and
                # there is no need to pass it to the `accelerator.prepare` call
                self.ref_model = self.ref_model.to(self.accelerator.device)

            # this hack seems to be needed for DS stage 3 to work
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3:
                self.model.train()
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        # init the current step
        self.current_step = 0

        # init variables for pushing model to hub
        if config.push_to_hub_if_best_kwargs:
            if "repo_id" not in config.push_to_hub_if_best_kwargs:
                raise ValueError(
                    "You have to specify repo_id in order to push the model to the hub!"
                )
            self.push_to_hub_kwargs = config.push_to_hub_if_best_kwargs
            self.compare_step = 0
            self.highest_reward = torch.tensor(-float("inf"))

        # post process for PP
        if not getattr(self.model, "is_sequential_parallel", False):
            self.current_device = self.accelerator.device
        else:
            self.current_device = torch.device("cuda:0")

        PPODecorators.optimize_cuda_cache = self.config.optimize_cuda_cache

    def _filter_kwargs(self, kwargs, target_func):
        """
        filter the keyword arguments that are supported by the target function.

        Args:
            kwargs (dict):
                Keyword arguments
            target_func (function):
                Target function
        """
        return {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(target_func).parameters.keys()
        }

    def prepare_dataloader(
        self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None
    ):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    # Adapted from transformers.Trainer._set_signature_columns_if_needed
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # label => sentiment | we need query and response for logging purpose
            self._signature_columns += list(set(["label", "query", "response"]))

    # Adapted from transformers.Trainer._remove_unused_columns
    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"],
                columns=columns,
                format_kwargs=dataset.format["format_kwargs"],
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`batch_size`, `seq_len`) containing query tokens.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """

        if isinstance(query_tensor, List):
            return self._generate_batched(
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )

        else:
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()
            response = self.accelerator.unwrap_model(self.model).generate(
                input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
            )

            if not return_prompt and not self.is_encoder_decoder:
                return response[:, query_tensor.shape[0] :]
            return response

    def _generate_batched(
        self,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        **generation_kwargs,
    ):
        outputs = []

        padding_side_default = self.tokenizer.padding_side
        if not self.is_encoder_decoder:
            self.tokenizer.padding_side = "left"

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {"input_ids": batch, "attention_mask": batch_mask}

            padded_inputs = self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=None,
                pad_to_multiple_of=pad_to_multiple_of,
                return_tensors="pt",
            ).to(self.current_device)

            generations = self.accelerator.unwrap_model(self.model).generate(
                **padded_inputs, **generation_kwargs
            )

            for generation, mask in zip(generations, padded_inputs["attention_mask"]):
                if not self.is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not return_prompt and not self.is_encoder_decoder:
                    output = output[(mask).sum() :]  # remove prompt
                outputs.append(output)

        self.tokenizer.padding_side = padding_side_default
        return outputs

    def _step_safety_checker(
        self,
        batch_size: int,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
        Returns:
            `tuple`: The input processed data.
        """
        for name, tensor_list in zip(
            ["queries", "responses", "scores"], [queries, responses, scores]
        ):
            if not isinstance(tensor_list, list):
                raise ValueError(
                    f"{name} must be a list of tensors - got {type(tensor_list)}"
                )
            if not isinstance(tensor_list[0], torch.Tensor):
                raise ValueError(
                    f"Elements in {name} must be tensors - got {type(tensor_list[0])}"
                )
            if batch_size is not None and len(tensor_list) != batch_size:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match number of examples - but"
                    f" got {len(tensor_list)} for: {name}"
                )

        # add queries, scores and responses on the correct device
        queries = [tensor.to(self.current_device) for tensor in queries]
        responses = [tensor.to(self.current_device) for tensor in responses]
        scores = [tensor.to(self.current_device) for tensor in scores]

        # squeeze scores if needed
        for i, score in enumerate(scores):
            if score.dim() > 1:
                raise ValueError(
                    f"Scores must be 1-dimensional - got {score.dim()} for {score}"
                )
            elif score.dim() == 1:
                scores[i] = score.squeeze()

        return queries, responses, scores

    @PPODecorators.empty_cuda_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        """
        Run a PPO optimisation step given a list of queries, model responses, and rewards.

        Args:
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.

        Returns:
            `dict[str, Any]`: A summary of the training statistics
        """
        # tqdm.write("we made it inside step")
        # print_memory_utilization()
        bs = self.config.batch_size

        queries, responses, scores = self._step_safety_checker(
            bs, queries, responses, scores
        )

        # if we want to push best model to the hub
        if hasattr(self, "highest_reward"):
            if self.compare_step % self.config.compare_steps == 0:
                curr_mean_reward = torch.tensor(scores).mean()
                # if the best reward ever seen
                if curr_mean_reward > self.highest_reward:
                    self.highest_reward = curr_mean_reward
                    # push model to hub
                    self.push_to_hub(**self.push_to_hub_kwargs)
            self.compare_step += 1

        timing = dict()
        t0 = time.time()

        t = time.time()

        model_inputs = self.prepare_model_inputs(queries, responses)
        # tqdm.write("We made it past prepare_model_inputs")

        if self.is_distributed:
            pad_first = self.tokenizer.padding_side == "left"

            model_inputs["input_ids"] = self.accelerator.pad_across_processes(
                model_inputs["input_ids"],
                dim=1,
                pad_index=self.tokenizer.pad_token_id,
                pad_first=pad_first,
            )
            model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
                model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
            )
            if self.is_encoder_decoder:
                model_inputs["decoder_input_ids"] = (
                    self.accelerator.pad_across_processes(
                        model_inputs["decoder_input_ids"],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                        pad_first=pad_first,
                    )
                )
                model_inputs["decoder_attention_mask"] = (
                    self.accelerator.pad_across_processes(
                        model_inputs["decoder_attention_mask"],
                        dim=1,
                        pad_index=0,
                        pad_first=pad_first,
                    )
                )

        model_inputs_names = list(model_inputs.keys())

        with torch.no_grad():
            all_logprobs, _, values, masks = self.batched_forward_pass(
                self.model, queries, responses, model_inputs
            )

            # for when the model is a peft model
            if self.is_peft_model and hasattr(
                self.accelerator.unwrap_model(self.model).pretrained_model,
                "disable_adapter",
            ):
                with self.accelerator.unwrap_model(
                    self.model
                ).pretrained_model.disable_adapter():
                    ref_logprobs, _, _, _ = self.batched_forward_pass(
                        self.model, queries, responses, model_inputs
                    )
            elif self.is_peft_model and not hasattr(
                self.model.pretrained_model, "disable_adapter"
            ):
                raise ValueError(
                    "You are using a `peft` version that does not support"
                    " `disable_adapter`. Please update your `peft` version to the"
                    " latest version."
                )

            else:
                ref_logprobs, _, _, _ = self.batched_forward_pass(
                    self.ref_model, queries, responses, model_inputs
                )

        timing["time/ppo/forward_pass"] = time.time() - t

        t = time.time()
        rewards, whitened_rewards, non_score_reward, kl = self.compute_rewards(
            scores, all_logprobs, ref_logprobs, masks
        )
        timing["time/ppo/compute_rewards"] = time.time() - t

        # upcast to float32 to avoid dataset issues
        mini_batch_dict = {
            "queries": queries,
            "responses": responses,
            "logprobs": all_logprobs.to(torch.float32),
            "values": values.to(torch.float32),
            "rewards": rewards,
            "whitened_rewards": whitened_rewards,
            "masks": masks,
            "kl": kl,
        }

        def collator(data):
            return_dict = dict()
            for key in data[0]:
                if key in ["queries", "responses"]:
                    return_dict[key] = [d[key] for d in data]
                else:
                    return_dict[key] = torch.stack([d[key] for d in data]).to(
                        self.current_device
                    )
            return return_dict

        mini_batch_dict.update(model_inputs)
        mini_batch_data = Dataset.from_dict(mini_batch_dict)
        mini_batch_data.set_format("torch")
        mini_batch_dataloader = torch.utils.data.DataLoader(
            mini_batch_data,
            batch_size=self.config.mini_batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        t = time.time()
        all_stats = []
        early_stop = False
        for _ in range(self.config.ppo_epochs):
            if early_stop:
                break
            for batch in mini_batch_dataloader:
                with self.accelerator.accumulate(self.model):
                    model_inputs = {k: batch[k] for k in model_inputs_names}
                    logprobs, logits, vpreds, _ = self.batched_forward_pass(
                        self.model,
                        batch["queries"],
                        batch["responses"],
                        model_inputs,
                        return_logits=True,
                    )

                train_stats = self.train_minibatch(
                    batch["logprobs"],
                    batch["values"],
                    batch["whitened_rewards"],
                    logprobs,
                    logits,
                    vpreds,
                    batch["masks"],
                )

                all_stats.append(train_stats)

                if self.config.early_stopping:
                    policykl = train_stats["policy/policykl"]
                    early_stop = self._early_stop(policykl)
                    if early_stop:
                        break

        timing["time/ppo/optimize_step"] = time.time() - t

        t = time.time()
        train_stats = stack_dicts(all_stats)

        # reshape advantages/ratios such that they are not averaged.
        train_stats["policy/advantages"] = torch.flatten(
            train_stats["policy/advantages"]
        ).unsqueeze(0)
        train_stats["policy/advantages"] = torch.nan_to_num(
            train_stats["policy/advantages"], WANDB_PADDING
        )
        train_stats["policy/ratio"] = torch.flatten(
            train_stats["policy/ratio"]
        ).unsqueeze(0)

        stats = self.record_step_stats(
            scores=scores,
            logprobs=all_logprobs,
            ref_logprobs=ref_logprobs,
            non_score_reward=non_score_reward,
            train_stats=train_stats,
            kl_coef=self.kl_ctl.value,
            masks=masks,
            queries=queries,
            responses=responses,
        )
        # Gather/Reduce stats from all processes
        if self.is_distributed:
            stats = self.gather_stats(stats)
        stats = stats_to_np(stats)
        timing["time/ppo/calc_stats"] = time.time() - t
        stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

        # Update the KL control - multiply the batch_size by the number of processes
        self.kl_ctl.update(
            stats["objective/kl"],
            self.config.batch_size * self.accelerator.num_processes,
        )

        # Log the total ppo time
        timing["time/ppo/total"] = time.time() - t0
        stats.update(timing)

        # post-process stats for tensorboard and other loggers
        if self.config.log_with != "wandb":
            stats = convert_to_scalar(stats)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # print_memory_utilization()

        return stats

    def _early_stop(self, policykl):
        r"""
        Handles the early stopping logic. If the policy KL is greater than the target KL, then the gradient is zeroed and
        the optimization step is skipped.
        This also handles the multi-gpu case where the policy KL is averaged across all processes.

        Args:
            policy_kl (torch.Tensor):
                the policy KL

        Returns:
            `bool`: whether to early stop or not
        """
        early_stop = False
        if not self.config.early_stopping:
            return early_stop

        if not self.is_distributed and policykl > 1.5 * self.config.target_kl:
            self.optimizer.zero_grad()
            early_stop = True
        elif self.is_distributed:
            import torch.distributed as dist

            # Wait for all processes to finish
            dist.barrier()

            # all gather the policykl
            dist.all_reduce(policykl, dist.ReduceOp.SUM)
            policykl /= self.accelerator.num_processes

            if policykl > 1.5 * self.config.target_kl:
                self.optimizer.zero_grad()
                early_stop = True
        return early_stop

    def gather_stats(self, stats):
        """
        Gather stats from all processes. Useful in the context of distributed training.

        Args:
            stats (dict[str, Any]):
            a dictionary of stats to be gathered. The stats should contain torch tensors.

        Returns:
            `dict[str, Any]`: A dictionary of stats with the tensors gathered.
        """
        import torch.distributed as dist

        # Wait for all processes to finish
        dist.barrier()

        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                dist.all_reduce(v, dist.ReduceOp.SUM)
                v /= self.accelerator.num_processes
            stats[k] = v
        return stats

    def prepare_model_inputs(self, queries: torch.Tensor, responses: torch.Tensor):
        if self.is_encoder_decoder:
            input_data = self.data_collator(
                [
                    {"input_ids": q, "attention_mask": torch.ones_like(q)}
                    for q in queries
                ]
            ).to(self.current_device)

            decoder_inputs = self.data_collator(
                [
                    {"input_ids": r, "attention_mask": torch.ones_like(r)}
                    for r in responses
                ]
            ).to(self.current_device)

            input_data["decoder_input_ids"] = decoder_inputs["input_ids"]
            input_data["decoder_attention_mask"] = decoder_inputs["attention_mask"]

        else:
            input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]
            input_data = self.data_collator(
                [
                    {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                    for ids in input_ids
                ]
            ).to(self.current_device)

        input_data.pop("labels", None)  # we don't want to compute LM losses

        return input_data

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
            return_logits (`bool`, *optional*, defaults to `False`):
                Whether to return all_logits. Set to `False` if logits are not needed to reduce memory consumption.
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses,
                    shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            input_kwargs = {
                key: value[i * fbs : (i + 1) * fbs]
                for key, value in model_inputs.items()
            }
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            # tqdm.write(f"Batch {i + 1}/{int(bs / fbs)}")
            # print_memory_utilization()
            logits, _, values = model(**input_kwargs)
            # tqdm.write(f"Logits shape: {logits.shape}. Made it pasat running model")
            # print_memory_utilization()

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(fbs):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])

                if len(logprobs[j, start:end]) < 2:
                    raise ValueError(
                        "Responses are too short. Make sure they are at least 4 tokens"
                        " long."
                    )

                masks[j, :start] = 0
                masks[j, end:] = 0

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        # tqdm.write("Batched forward pass done.")
        # print_memory_utilization()

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @PPODecorators.empty_cuda_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Train one PPO minibatch

        Args:
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape [batch_size, response_length]
            values (`torch.FloatTensor`):
                Values of the value head, shape [batch_size, response_length]
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape [batch_size, response_length]
            query (`torch.LongTensor`):
                Encoded queries, shape [batch_size, query_length]
            response (`torch.LongTensor`):
                Encoded responses, shape [batch_size, response_length]
            model_input (`torch.LongTensor`):
                Concatenated queries and responses, shape [batch_size, query_length+response_length]

        Returns:
            train_stats (dict[str, `torch.Tensor`]):
                Dictionary of training statistics
        """
        # tqdm.write("Before train minibatch starts")
        # print_memory_utilization()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, rewards, logits, vpreds, logprobs, mask
        )
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)

        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.config.max_grad_norm,
            )

        t = time.time()
        self.optimizer.step()
        # tqdm.write("Train Minibatch step done.")
        # print_memory_utilization()
        train_stats["time/ppo/optimizer_step"] = torch.Tensor([time.time() - t]).to(
            self.current_device
        )
        return train_stats

    def compute_rewards(
        self,
        scores: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        ref_logprobs: torch.FloatTensor,
        masks: torch.LongTensor,
    ):
        """
        Compute per token rewards from scores and KL-penalty.

        Args:
            scores (`torch.FloatTensor`):
                Scores from the reward model, shape (`batch_size`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            ref_logprobs (`torch.FloatTensor`):
                Log probabilities of the reference model, shape (`batch_size`, `response_length`)
        """
        rewards, non_score_rewards, kls = [], [], []
        for score, logprob, ref_logprob, mask in zip(
            scores, logprobs, ref_logprobs, masks
        ):
            # compute KL penalty (from difference in logprobs)
            kl = logprob - ref_logprob
            if self.config.clip_kl:
                kl = torch.maximum(kl, torch.tensor(0))
            non_score_reward = -self.kl_ctl.value * kl
            non_score_rewards.append(non_score_reward)
            reward = non_score_reward.clone()
            last_non_masked_index = mask.nonzero()[-1]

            # reward is preference model score + KL penalty
            reward[last_non_masked_index] += score
            rewards.append(reward)
            kls.append(kl)
        # optionally whitten rewards
        rewards = torch.stack(rewards)
        whitened_rewards = rewards
        if self.config.whiten_rewards:
            whitened_rewards = masked_whiten(rewards, masks, shift_mean=False, verbose=True)
        return rewards, whitened_rewards, torch.stack(non_score_rewards), torch.stack(kls)

    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        """
        Calculate policy and value losses.

        Args:
            old_logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
            values (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            rewards (`torch.FloatTensor`):
                Rewards from the reward model, shape (`batch_size`, `response_length`)
            logits (`torch.FloatTensor`):
                Logits of the model, shape (`batch_size`, `response_length`, `vocab_size`)
            v_pred (`torch.FloatTensor`):
                Values of the value head, shape (`batch_size`, `response_length`)
            logprobs (`torch.FloatTensor`):
                Log probabilities of the model, shape (`batch_size`, `response_length`)
        """
        lastgaelam = 0
        advantages_reversed = []
        gen_len = rewards.shape[-1]

        values = values * mask
        rewards = rewards * mask
        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.config.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.config.gamma * self.config.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = masked_whiten(advantages, mask)
        advantages = advantages.detach()

        vpredclipped = clip_by_value(
            vpreds,
            values - self.config.cliprange_value,
            values + self.config.cliprange_value,
        )

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).double(), mask)

        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange
        )

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).double(), mask)

        loss = pg_loss + self.config.vf_coef * vf_loss

        entropy = masked_mean(entropy_from_logits(logits), mask)
        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)
        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(
                policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()
            ),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss, flatten_dict(stats)

    def record_step_stats(self, kl_coef: float, **data):
        """
        Record training step statistics.


        Args:
            kl_coef (`float`):
                KL coefficient
            data (`dict`):
                Dictionary of training step data

        Returns:
            stats (`dict`):
                Dictionary of training step statistics
        """
        mask = data.pop("masks")

        kl_list = ((data["logprobs"] - data["ref_logprobs"]) * mask).sum(axis=-1)
        mean_kl = kl_list.mean()
        mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()

        mean_non_score_reward = masked_mean(
            data["non_score_reward"], mask
        )  # non_score_reward is size `batch_size`, `response_length`
        mean_scores = torch.stack(data["scores"]).mean()  # scores is size `batch_size`
        std_scores = torch.stack(data["scores"]).std()

        if mean_kl.item() < 0.0:
            # warn users
            warnings.warn(
                f"KL divergence is starting to become negative: {mean_kl.item():.2f} -"
                " this might be a precursor for failed training. sometimes this"
                " happens because the generation kwargs are not correctly set. Please"
                " make sure that the generation kwargs are set correctly, or review"
                " your training hyperparameters."
            )

        stats = {
            "objective/kl": mean_kl,
            "objective/kl_dist": kl_list,
            "objective/logprobs": data["logprobs"],
            "objective/ref_logprobs": data["ref_logprobs"],
            "objective/kl_coef": kl_coef,
            "objective/entropy": mean_entropy,
            "ppo/mean_non_score_reward": mean_non_score_reward,
            "ppo/mean_scores": mean_scores,
            "ppo/std_scores": std_scores,
        }

        # Log text properties
        query_lens = torch.tensor(
            [len(query) for query in data["queries"]], dtype=torch.float
        )
        response_lens = torch.tensor(
            [len(response) for response in data["responses"]], dtype=torch.float
        )

        stats["tokens/queries_len_mean"] = torch.mean(query_lens).cpu().numpy().item()
        stats["tokens/queries_len_std"] = torch.std(query_lens).cpu().numpy().item()
        stats["tokens/queries_dist"] = query_lens.cpu().numpy()
        stats["tokens/responses_len_mean"] = (
            torch.mean(response_lens).cpu().numpy().item()
        )
        stats["tokens/responses_len_std"] = (
            torch.std(response_lens).cpu().numpy().item()
        )
        stats["tokens/responses_dist"] = response_lens.cpu().numpy()

        for k, v in data["train_stats"].items():
            stats[f"ppo/{k}"] = torch.mean(v, axis=0)
        stats["ppo/val/var_explained"] = (
            1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
        )
        return stats

    def log_stats(
        self,
        stats: dict,
        batch: dict,
        rewards: List[torch.FloatTensor],
        test_rewards: Optional[List[torch.FloatTensor]] = None,
    ):
        """
        A function that logs all the training stats. Call it at the end of each epoch.

        Args:
            stats (dict[str, Any]):
                A dictionary of training stats.
            batch (dict[str, Any]):
                A dictionary of batch data, this contains the queries and responses.
            rewards (`List[torch.FloatTensor]`):
                A tensor of rewards.
        """
        # Log only if we are in the main process
        if self.accelerator.is_main_process:
            test_rewards = test_rewards.to(torch.float32)
            logs = {}

            # Log stats
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards).to(self.current_device)

            if "query" not in batch.keys() and "response" not in batch.keys():
                # warn the user that the game logs will not be logged
                warnings.warn(
                    "The game logs will not be logged because the batch does not"
                    " contain the keys 'query' and 'response'. "
                )
            elif self.config.log_with == "wandb":
                import wandb

                table_rows = [
                    list(r)
                    for r in zip(
                        batch["query"],
                        batch["response"],
                        rewards.cpu().tolist(),
                        (
                            test_rewards.cpu().tolist()
                            if test_rewards is not None
                            else [None] * len(batch["query"])
                        ),
                    )
                ]
                logs.update(
                    {
                        "game_log": wandb.Table(
                            columns=["query", "response", "reward", "test_reward"],
                            rows=table_rows,
                        )
                    }
                )
            # All reduce rewards if distributed
            if self.is_distributed:
                import torch.distributed as dist

                dist.barrier()

                dist.all_reduce(rewards, op=torch.distributed.ReduceOp.SUM)
                rewards /= self.accelerator.num_processes

            logs.update(stats)

            # manually cast in fp32 for bf16 torch tensors
            for k, v in logs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    logs[k] = v.float()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/reward_dist"] = rewards.cpu().numpy()

            logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
            logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
            logs["env/reward_dist"] = rewards.cpu().numpy()

            logs["env/test_reward_mean"] = torch.mean(test_rewards).cpu().numpy().item()
            logs["env/test_reward_std"] = torch.std(test_rewards).cpu().numpy().item()
            logs["env/test_reward_dist"] = test_rewards.cpu().numpy()

            logs["env/test_reward_mean"] = torch.mean(test_rewards).cpu().numpy().item()
            logs["env/test_reward_std"] = torch.std(test_rewards).cpu().numpy().item()
            logs["env/test_reward_dist"] = test_rewards.cpu().numpy()

            if self.config.log_with == "tensorboard":
                # update the current step
                self.current_step += 1

            self.accelerator.log(
                logs,
                step=(
                    self.current_step if self.config.log_with == "tensorboard" else None
                ),
            )

        else:
            if self.is_distributed:
                import torch.distributed as dist

                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards).to(self.current_device)

                dist.barrier()
                dist.all_reduce(rewards, op=torch.distributed.ReduceOp.SUM)

    def create_model_card(
        self, path: str, model_name: Optional[str] = "TRL Model"
    ) -> None:
        """Creates and saves a model card for a TRL model.

        Args:
            path (`str`): The path to save the model card to.
            model_name (`str`, *optional*): The name of the model, defaults to `TRL Model`.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        user = whoami()["name"]
        model_card_content = MODEL_CARD_TEMPLATE.format(
            model_name=model_name, model_id=f"{user}/{path}"
        )
        with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)

    def _save_pretrained(self, save_directory: str) -> None:
        self.accelerator.unwrap_model(self.model).save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        self.create_model_card(save_directory)
