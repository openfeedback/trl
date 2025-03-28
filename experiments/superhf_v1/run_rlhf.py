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
import re

from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer, HfArgumentParser, pipeline, get_scheduler


from torchtyping import TensorType

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model, set_seed
from trl.core import respond_to_batch, LengthSampler

from datasets import load_dataset, Dataset

T = TypeVar("T")

import wandb

import os

import constants
from utils import separate_prompt_from_completion

WANDB_ENTITY_NAME = "stanfordaialignment"
WANDB_PROJECT_NAME = "rlhf-trl-v1"


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

class ListDataset(Dataset):
    """A Torch dataset that wraps a list of data."""

    def __init__(self, data):
        self.data = data

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
    config: Optional[str] = field(
        # Get file pa`th relative to this file
        default=os.path.join(os.path.dirname(__file__), "configs", "rlhf_config.yaml"),
        metadata={"help": "The name of the Weights and Biases config to use."}
    )
    # model_name: Optional[str] = field(default="EleutherAI/gpt-neo-125M", metadata={"help": "the model name"})
    # log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    # learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    # mini_batch_size: Optional[int] = field(default=16, metadata={"help": "the PPO minibatch size"})
    # batch_size: Optional[int] = field(default=256, metadata={"help": "the batch size"})
    # gradient_accumulation_steps: Optional[int] = field(
    #     default=1, metadata={"help": "the number of gradient accumulation steps"}
    # )
    # dataset_names: Optional[List[str]] = field(
    #     default_factory=lambda: ["anthropic-red-team"],
    #     metadata={"help": "the dataset name(s) to use"}
    # )
    # debug_max_prompts: Optional[int] = field(
    #     default=0,
    #     metadata={"help": "the maximum number of prompts to use for debugging"}
    # )
    notes: Optional[str] = field(
        default="",
        metadata={"help": "notes to add to the wandb run"}
    )

def parse_args():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    return script_args

def build_dataset(dataset_names, tokenizer, max_prompt_char_length=1024, debug_max_prompts=0, conversation_prompt=""):
    """
    Currentlty we don't use the tokenizer becauses the internal trainer 
    for some reason throws away the tokenized exmples.s

    Returns:
        a pytorch dataset that implements the __getitem__ and __len__ methods.
        PPO trainer converts this to a pytorch dataloader.
        torch.utils.data.Dataset
    """
    prompts: list[str] = []
    for dataset in dataset_names:
        prompts.extend(get_superhf_prompts(dataset))
    
    random.shuffle(prompts)
    print(f"Loaded {len(prompts)} prompts from {dataset}")
    # Filter out prompts that are too long
    old_prompt_count = len(prompts)
    prompts = [
        conversation_prompt + prompt
        for prompt in prompts
        if len(conversation_prompt + prompt) < max_prompt_char_length
    ]
    print(
        f"Filtered {old_prompt_count - len(prompts)} prompts over "
        f"{max_prompt_char_length} chars."
    )

    # Only load the first section of prompts
    if debug_max_prompts != 0:
        prompts = prompts[: debug_max_prompts]

    print(f"Loaded {len(prompts)} prompts.")

    def tokenize(sample):
        dictionized_example = {}
        # dictionized_example["input_ids"] = tokenizer.encode(sample)
        dictionized_example["query"] = sample # tokenizer.decode(dictionized_example["input_ids"])
        return dictionized_example
    
    prompts_2 = [tokenize(prompt) for prompt in prompts]
    prompts_3 = {"query": [d["query"] for d in prompts_2] }
    dataset = Dataset.from_dict(prompts_3)
    dataset.set_format(type="torch")
    return dataset

def main():
    script_args = parse_args()
    wandb.init(
        entity=WANDB_ENTITY_NAME,
        project=WANDB_PROJECT_NAME,
        notes=script_args.notes,
        save_code=True,
        config=script_args.config,
    )

    ppo_config = PPOConfig(
        model_name=wandb.config.model_name,
        learning_rate=wandb.config.learning_rate,
        log_with=wandb.config.log_with,
        mini_batch_size=wandb.config.mini_batch_size,
        batch_size=wandb.config.batch_size,
        gradient_accumulation_steps=wandb.config.gradient_accumulation_steps,
        seed=66,
    )

    assert ppo_config.mini_batch_size <= ppo_config.batch_size
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
    model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(ppo_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name, padding_side='left')

    sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": wandb.config.batch_size}
    
    # set seed before initializing value head for deterministic eval
    set_seed(ppo_config.seed)

    dataset = build_dataset(wandb.config.dataset_names, tokenizer, max_prompt_char_length=wandb.config.max_prompt_char_length, debug_max_prompts=wandb.config.debug_max_prompts, conversation_prompt=wandb.config.conversation_prompt)    

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    scheduler = None
    optimizer = None
    if wandb.config.scheduler_name != "":
        num_training_steps = len(dataset) // (wandb.config.batch_size * wandb.config.gradient_accumulation_steps)
        optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=wandb.config.learning_rate
            )
        scheduler = get_scheduler(wandb.config.scheduler_name, optimizer=optimizer, num_warmup_steps=wandb.config.scheduler_warmup_steps, num_training_steps=num_training_steps)
    # create a ppo trainer config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
    # the dataset and collator get bundled in a data loader together. 
    ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer, lr_scheduler=scheduler) #, data_collator=collator)

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    # This pipelinle is for hte reward model
    sentiment_pipe = pipeline(model=wandb.config.reward_model_name, device=device)
    print(f"The device is {device}")

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

    # input_size = LengthSampler(input_min_text_length, input_max_text_length)

    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)
    tokenizer.pad_token = tokenizer.eos_token
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
        query_tensors = [tokenizer(q, return_tensors="pt")["input_ids"].squeeze().to(device) for q in batch["query"]]
        
        # Get response from the model
        response_tensors = []
        for query in query_tensors:
            # gen_len = output_length_sampler()
            # generation_kwargs["max_new_tokens"] = gen_len
            generation_kwargs["max_new_tokens"] = wandb.config.max_new_tokens
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze())
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        batch["response"] = trim_generations(batch["response"])

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        if len(pipe_outputs[0]) > 1:
            print(f"len of output is {len(pipe_outputs[0])}, so maybe it should be output[1]['score']?")
        rewards = [torch.tensor(output[0]["score"]) for output in pipe_outputs]

        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        if len(wandb.config.hub_repo_id) > 0 and (epoch == len(ppo_trainer.dataloader) - 1 or (epoch > 0 and epoch % wandb.config.save_every == 0)):
            run_name = wandb.run.name
            tqdm.write(f"Pushing model and tokenizer to the Hub! Location: {wandb.config.hub_repo_id}")
            ppo_trainer.model.push_to_hub(
                repo_id=wandb.config.hub_repo_id,
                        commit_message=(
                            f"Upload model from batch {epoch}, run {run_name}"
                        )
            )
            ppo_trainer.tokenizer.push_to_hub(
                repo_id=wandb.config.hub_repo_id,
                        commit_message=(
                            f"Upload tokenizer from batch {epoch}, run {run_name}"
                        )
            )


def trim_generations(raw_completions: list[str]) -> list[str]:
    original_length = len(raw_completions)
    prompts_and_completions = [
        separate_prompt_from_completion(completion)
        for completion in raw_completions
    ]
    trimmed_completions: list[str] = []
    model_completion_lengths: list[int] = []
    for prompt, completion in prompts_and_completions:
        stripped_completion = re.split(
            constants.PROMPT_DELIMITER_REGEX_COMPLEX, completion, maxsplit=1
        )[0].strip()
        # if stripped_completion == "":
        #     continue
        trimmed_completions.append(prompt + " " + stripped_completion)
        model_completion_lengths.append(len(stripped_completion))

    assert len(trimmed_completions) == original_length, "Trimmed completions should have the same length as the original completions."
    return trimmed_completions

if __name__ == "__main__":
    main()