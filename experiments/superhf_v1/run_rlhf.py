"""
Script for running RLHF to compare with SuperHF.

Example usage:
    python run_rlhf.py \
        --model_name "EleutherAI/gpt-neo-125M" \
        --mini_batch_size 1 \
        --log_with wandb
"""

# imports
from dataclasses import dataclass, field
from typing import Optional, Iterator, TypeVar, List
import random

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from transformers import AutoTokenizer, HfArgumentParser, pipeline


from torchtyping import TensorType

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model, set_seed
from trl.core import respond_to_batch, LengthSampler

from datasets import load_dataset

T = TypeVar("T")

WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "shf-iterative-v2"


def get_superhf_prompts(dataset_name: str, split: str = "train") -> list[str]:
    """
    Get a list of prompts from a dataset.
    Args:
        dataset_name: The name of the dataset to load. One of:
            - 'anthropic-red-team'
        split: The split of the dataset to load.
    Returns:
        A list of prompts.
    """
    # Load the appropriate dataset then convert to a list of prompts
    prompts: list[str] = []
    if dataset_name == "anthropic-red-team":
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="red-team-attempts",
            split=split,
            keep_in_memory=False,
        )
        prompts.extend(
            [
                dict(row)["transcript"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                for row in dataset
            ]
        )
    elif dataset_name == "openai/webgpt_comparisons":
        dataset = load_dataset(
            dataset_name,
            split=split,
            keep_in_memory=False,
        )
        prompts.extend(
            [
                "\n\nHuman: " + row["question"]["full_text"] + "\n\nAssistant:"
                for row in dataset
            ]
        )
    elif dataset_name == "anthropic-harmless-base":
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="harmless-base",
            split=split,
            keep_in_memory=False,
        )
        prompts.extend(
            [
                row["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                for row in dataset
            ]
        )
    elif dataset_name == "anthropic-helpful-base":
        dataset = load_dataset(
            "Anthropic/hh-rlhf",
            data_dir="helpful-base",
            split=split,
            keep_in_memory=False,
        )
        prompts.extend(
            [
                row["chosen"].split("\n\nAssistant:")[0] + "\n\nAssistant:"
                for row in dataset
            ]
        )
    elif dataset_name == "mock":
        prompts.extend(
            [
                f"{i}\n\nHuman: ...\n\nAssistant: Sphinx of black quartz, judge my vow."
                for i in range(50000)
            ]
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return prompts

class ListDataset(IterableDataset[T]):
    """A Torch dataset that wraps a list of data."""

    def __init__(self, data: list[T]):
        self.data = data

    def __iter__(self) -> Iterator[T]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> T:
        return self.data[index]
    
# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="EleutherAI/gpt-neo-125M", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=16, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    dataset_names: Optional[List[str]] = field(
        default_factory=lambda: ["anthropic-red-team"],
        metadata={"help": "the dataset name(s) to use"}
    )
    debug_max_prompts: Optional[int] = field(
        default=100,
        metadata={"help": "the maximum number of prompts to use for debugging"}
    )

def parse_args():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    return script_args

def build_dataset(dataset_names, max_prompt_char_length=1024, debug_max_prompts=0):
    """
    Returns:
        a pytorch dataset that implements the __getitem__ and __len__ methods.
        PPO trainer converts this to a pytorch dataloader.
    """
    prompts: list[str] = []
    for dataset in dataset_names:
        prompts.extend(get_superhf_prompts(dataset))
    
    random.shuffle(prompts)
    print(f"Loaded {len(prompts)} prompts from {dataset}")
    # Filter out prompts that are too long
    old_prompt_count = len(prompts)
    prompts = [
        prompt
        for prompt in prompts
        if len(prompt) < max_prompt_char_length
    ]
    print(
        f"Filtered {old_prompt_count - len(prompts)} prompts over "
        f"{max_prompt_char_length} chars."
    )

    # Only load the first section of prompts
    if debug_max_prompts != 0:
        prompts = prompts[: debug_max_prompts]

    print(f"Loaded {len(prompts)} prompts.")

    return ListDataset(prompts)

def generate_completions(
    self,
    minibatch_size: int,
    superbatch_prompts: list[str],
) -> list[TensorType["batch", "seq_len"]]:
    """
    Generate completions for the prompts in the superbatch.

    Args:
        minibatch_size: The minibatch size to use for generation.
        superbatch_prompts: The prompts in the superbatch.
    """
    self.training_args.minibatch_size_generating = minibatch_size

    tqdm.write(f"Trying generation with batch size {minibatch_size}")

    completion_dataloader = DataLoader(
        ListDataset(superbatch_prompts),
        batch_size=minibatch_size,
        collate_fn=self.collate_fn_lm_completions,
        pin_memory=True,
    )

    completions: list[TensorType["batch", "seq_len"]] = []
    with torch.no_grad():
        for minibatch in tqdm(
            completion_dataloader,
            desc="Generation",
            total=len(completion_dataloader),
        ):
            encodings = minibatch
            encodings.to(self.language_model.device)
            completions.extend(
                self.language_model.generate(
                    **encodings,
                    max_new_tokens=self.training_args.max_new_tokens,
                    temperature=self.training_args.temperature,
                    top_p=self.trainings_args.top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.language_tokenizer.pad_token_id,
                ).to("cpu")
            )
    return completions


def main():
    script_args = parse_args()
    ppo_config = PPOConfig(
        model_name=script_args.model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        seed=66,
    )
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    
    # set seed before initializing value head for deterministic eval
    set_seed(ppo_config.seed)

    # # encode a query
    # query_txt = "This morning I went to the "
    # query_tensor = tokenizer.encode(query_txt, return_tensors="pt")

    # # get model response
    # response_tensor  = respond_to_batch(model_ref, query_tensor)

    dataset = build_dataset(script_args.dataset_names, ppo_config.max_prompt_char_length, script_args.debug_max_prompts)


    # create a ppo trainer config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
    ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer, dataset=dataset, data_collator=collators)

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        # Get response from gpt2
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    ###########################################

    # define a reward for response
    # # (this could be any reward such as human feedback or output from another model)
    # reward = [torch.tensor(1.0)]

    # # train model for one step with ppo
    # train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], reward)
    # print(train_stats)


if __name__ == "__main__":
    main()