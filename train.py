"""
▪  Uses **MistralGRPOLoss** (Clip‑Higher, no KL) by default.
▪  Removes KL computations and reference‑model passes (speed‑up).
▪  Implements the two‑stage advantage normalisation exactly as in
   the prompt:
       1) group baseline     Â_i = r_i - μ_group
       2) batch std‑scaling  Â_norm = (Â_i - μ_batch) / σ_batch
▪  Filters out "non‑diverse" groups   (returns.std() == 0).
▪  Loss is averaged over all tokens via  masked_mean(..., dim=None).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
)

from loss import MistralGRPOLoss, GRPOLoss, masked_mean
from replay_buffer import ReplayBuffer, Experience, join_experience_batch


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
) -> tuple[LlamaForCausalLM, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: LlamaForCausalLM,
    tokenizer: Any,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """
    Generates `num_rollouts` completions for a single prompt and returns

        sequence_ids : [G, ≤max_len]     – full prompt + completion
        returns      : [G, 1]            – scalar rewards
        action_mask  : [G, L-1] bool     – indicates tokens *to* optimise
        completions  : list[str]         – raw decoded strings
    """
    model.eval()

    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to("cuda")

    # Duplicate prompt `num_rollouts` times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )
    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    pad_token_id = tokenizer.eos_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(
        **model_inputs, generation_config=generation_config
    )
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]  # shift – first token has no log‑prob

    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        answer_match = re.search(
            r"<answer>(.*?)</answer>", completion, flags=re.DOTALL
        )
        answer = answer_match.group(1) if answer_match else None
        reward = 0.0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01
        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def group_advantages(returns: torch.Tensor) -> torch.Tensor:
    """
    stage 1:
        Â_i = r_i − μ_group

    (No scaling by std here – that is done at mini‑batch level in train loop.)
    """
    return returns - returns.mean()


def sequence_log_probs_from_logits(
    logits: torch.Tensor, output_ids: torch.Tensor
) -> torch.Tensor:
    log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: LlamaForCausalLM,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes log π(model) for every *action* token.
    Output shape: [B, L-1]
    """
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def main() -> None:
    # Hyper‑parameters
    seed = 42
    wandb_project = None  # set to a string to enable logging
    device_index = 0
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    checkpoint_path = Path("./output")
    checkpoint_interval = 20

    train_batch_size = 16
    lr = 5e-6
    max_norm = 1.0  # gradient clipping

    # KL removed 
    kl_weight = 0.0

    # Clip‑Higher thresholds
    clip_eps_low = 0.20
    clip_eps_high = 0.28

    # Roll‑out parameters
    group_size = 12                # G generations per prompt
    rollouts_per_step = 32         # prompts per outer loop
    epochs_per_step = 1
    max_length = 1024
    top_p = 1.0
    temperature = 1.0

    device = torch.device("cuda", device_index)
    cpu_device = torch.device("cpu")
    init_rng(seed)

    # Load current model (trainable) refeerence model no longer required
    model, tokenizer = load_model(model_name, device_map=device)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = MistralGRPOLoss(eps_low=clip_eps_low, eps_high=clip_eps_high)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []
        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(questions, answers):
                (
                    sequence_ids,
                    returns,
                    action_mask,
                    _,
                ) = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                # Filter out non‑diverse groups  if all rewards identical
                if returns.std() < 1e-8:
                    continue

                advantages = group_advantages(returns)  # stage‑1 baseline
                attention_mask = sequence_ids != pad_token_id

                # log πθ_old  (policy at collection time = current model)
                old_log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )

                # Create experience – KL/ref logs removed set to None
                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=old_log_probs,
                    log_probs_ref=None,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=None,
                )
                replay_buffer.append(experience.to(cpu_device))
                rollout_returns.append(returns.cpu())

        torch.cuda.empty_cache()

        episode_return_sum = torch.stack(rollout_returns).sum()
        print(f"returns of step {k}: {episode_return_sum:.4f}")
        wandb.log({"returns": episode_return_sum})

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience = exp.to(device)
                optimizer.zero_grad()

                # log πθ  (current, potentially updated model)
                log_probs = sequences_log_probs(
                    model,
                    sequence_ids=exp.sequences,
                    attention_mask=exp.attention_mask,
                )

                adv_mean = exp.advantages.mean()
                adv_std = exp.advantages.std(unbiased=False) + 1e-8
                exp.advantages = (exp.advantages - adv_mean) / adv_std

                loss = objective(log_probs=log_probs, experience=exp)

                if not loss.isfinite():
                    print("Loss not finite – skipping backward.")
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)

                with torch.no_grad():
                    policy_entropy = -torch.mean(torch.exp(log_probs) * log_probs)
                    policy_kl = torch.mean(exp.action_log_probs - log_probs)
                    
                    reward_mean = exp.returns.mean()
                    reward_std = exp.returns.std()
                    reward_max = exp.returns.max()
                    reward_min = exp.returns.min()
                    
                    action_ratio = exp.action_mask.float().mean()

                wandb.log({
                    "grad_norm": grad_norm,
                    "loss": loss.item(),
                    "adv_mean": adv_mean.item(),
                    "adv_std": adv_std.item(),
                    "policy_entropy": policy_entropy.item(),
                    "policy_kl": policy_kl.item(),
                    "reward_mean": reward_mean.item(),
                    "reward_std": reward_std.item(),
                    "reward_max": reward_max.item(),
                    "reward_min": reward_min.item(),
                    "action_ratio": action_ratio.item(),
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "step": k * epochs_per_step + step_epoch
                })

                optimizer.step()

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()