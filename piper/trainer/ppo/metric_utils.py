from collections import defaultdict
from functools import partial
from typing import Any, Dict

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import bootstrap_metric, calc_maj_val


def _compute_multi_turn_response_info(batch: DataProto) -> Dict[str, Any]:
    """Compute response information for multi-turn conversations."""
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_attention_mask = batch.batch["attention_mask"][:, -response_length:]
    response_mask = batch.batch["loss_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_attention_mask.sum(-1).float()

    # Multi-turn specific masks
    tool_tokens = (response_mask == 0) & (response_attention_mask == 1)
    assistant_tokens = (response_attention_mask == 1) & (tool_tokens == 0)

    assistant_length = assistant_tokens.sum(-1).float()
    tool_length = tool_tokens.sum(-1).float()

    raw_responses = batch.non_tensor_batch["raw_responses"]
    num_messages = torch.tensor([len(messages) for messages in raw_responses], dtype=torch.float32)
    num_assistant_messages = torch.tensor([sum(1 for message in messages if message["role"] == "assistant") for messages in raw_responses], dtype=torch.float32)
    num_tool_messages = torch.tensor([sum(1 for message in messages if message["role"] == "tool") for messages in raw_responses], dtype=torch.float32)

    return dict(
        prompt_length=prompt_length,
        response_length=response_length,
        assistant_tokens=assistant_tokens,
        tool_tokens=tool_tokens,
        assistant_length=assistant_length,
        tool_length=tool_length,
        num_messages=num_messages,
        num_assistant_messages=num_assistant_messages,
        num_tool_messages=num_tool_messages,
    )


def compute_multi_turn_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """Compute comprehensive metrics for multi-turn conversations."""
    response_info = _compute_multi_turn_response_info(batch)

    max_prompt_length = response_info["prompt_length"].max().item()

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    # Use loss_mask for valid tokens (excludes tool responses from loss calculation)
    max_response_length = batch.batch["responses"].shape[-1]
    loss_response_mask = batch.batch["loss_mask"][:, -max_response_length:].bool()
    valid_adv = torch.masked_select(advantages, loss_response_mask)
    valid_returns = torch.masked_select(returns, loss_response_mask)

    metrics = {
        # Overall advantages and returns (based on loss_mask)
        "multi-turn_critic/advantages/mean": torch.mean(valid_adv).detach().item() if len(valid_adv) > 0 else 0.0,
        "multi-turn_critic/advantages/max": torch.max(valid_adv).detach().item() if len(valid_adv) > 0 else 0.0,
        "multi-turn_critic/advantages/min": torch.min(valid_adv).detach().item() if len(valid_adv) > 0 else 0.0,
        "multi-turn_critic/returns/mean": torch.mean(valid_returns).detach().item() if len(valid_returns) > 0 else 0.0,
        "multi-turn_critic/returns/max": torch.max(valid_returns).detach().item() if len(valid_returns) > 0 else 0.0,
        "multi-turn_critic/returns/min": torch.min(valid_returns).detach().item() if len(valid_returns) > 0 else 0.0,
        # Prompt length metrics
        "multi-turn_length/prompt_length/mean": torch.mean(response_info["prompt_length"]).detach().item(),
        "multi-turn_length/prompt_length/max": torch.max(response_info["prompt_length"]).detach().item(),
        "multi-turn_length/prompt_length/min": torch.min(response_info["prompt_length"]).detach().item(),
        "multi-turn_length/prompt_length/clip_ratio": torch.mean(torch.eq(response_info["prompt_length"], max_prompt_length).float()).detach().item(),
        # Multi-turn specific metrics - Response length
        "multi-turn_length/response_length/mean": torch.mean(response_info["response_length"]).detach().item(),
        "multi-turn_length/response_length/max": torch.max(response_info["response_length"]).detach().item(),
        "multi-turn_length/response_length/min": torch.min(response_info["response_length"]).detach().item(),
        "multi-turn_length/response_length/total": torch.sum(response_info["response_length"]).detach().item(),
        # Multi-turn specific metrics - Assistant tokens
        "multi-turn_length/assistant_length/mean": torch.mean(response_info["assistant_length"]).detach().item(),
        "multi-turn_length/assistant_length/max": torch.max(response_info["assistant_length"]).detach().item(),
        "multi-turn_length/assistant_length/min": torch.min(response_info["assistant_length"]).detach().item(),
        "multi-turn_length/assistant_length/total": torch.sum(response_info["assistant_length"]).detach().item(),
        # Multi-turn specific metrics - Tool tokens
        "multi-turn_length/tool_length/mean": torch.mean(response_info["tool_length"]).detach().item(),
        "multi-turn_length/tool_length/max": torch.max(response_info["tool_length"]).detach().item(),
        "multi-turn_length/tool_length/min": torch.min(response_info["tool_length"]).detach().item(),
        "multi-turn_length/tool_length/total": torch.sum(response_info["tool_length"]).detach().item(),
        # Multi-turn specific metrics - Token distribution ratios
        "multi-turn/token_ratio/assistant": torch.sum(response_info["assistant_length"]).detach().item() / torch.sum(response_info["response_length"]).detach().item() if torch.sum(response_info["response_length"]).detach().item() > 0 else 0.0,
        "multi-turn/token_ratio/tool": torch.sum(response_info["tool_length"]).detach().item() / torch.sum(response_info["response_length"]).detach().item() if torch.sum(response_info["response_length"]).detach().item() > 0 else 0.0,
        # Multi-turn specific metrics - Number of messages
        "multi-turn/num_messages/mean": torch.mean(response_info["num_messages"]).detach().item(),
        "multi-turn/num_messages/max": torch.max(response_info["num_messages"]).detach().item(),
        "multi-turn/num_messages/min": torch.min(response_info["num_messages"]).detach().item(),
        "multi-turn/num_messages/total": torch.sum(response_info["num_messages"]).detach().item(),
        # assistant messages
        "multi-turn/num_assistant_messages/mean": torch.mean(response_info["num_assistant_messages"]).detach().item(),
        "multi-turn/num_assistant_messages/max": torch.max(response_info["num_assistant_messages"]).detach().item(),
        "multi-turn/num_assistant_messages/min": torch.min(response_info["num_assistant_messages"]).detach().item(),
        "multi-turn/num_assistant_messages/total": torch.sum(response_info["num_assistant_messages"]).detach().item(),
        # tool messages
        "multi-turn/num_tool_messages/mean": torch.mean(response_info["num_tool_messages"]).detach().item(),
        "multi-turn/num_tool_messages/max": torch.max(response_info["num_tool_messages"]).detach().item(),
        "multi-turn/num_tool_messages/min": torch.min(response_info["num_tool_messages"]).detach().item(),
        "multi-turn/num_tool_messages/total": torch.sum(response_info["num_tool_messages"]).detach().item(),
    }

    # Add critic-specific metrics if available
    if use_critic and "values" in batch.batch:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, loss_response_mask)

        if len(valid_values) > 0 and len(valid_returns) > 0:
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)

            metrics.update(
                {
                    "multi-turn_critic/values/mean": torch.mean(valid_values).detach().item(),
                    "multi-turn_critic/values/max": torch.max(valid_values).detach().item(),
                    "multi-turn_critic/values/min": torch.min(valid_values).detach().item(),
                    "multi-turn_critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
                }
            )

    return metrics


def compute_multi_turn_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """Wrapper function that maintains compatibility with existing verl interface."""
    return compute_multi_turn_metrics(batch, use_critic)


def process_validation_metrics_custom(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], n: int = 1, seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """Custom process validation metrics that groups by original example ID.

    This function groups every n consecutive samples together as generations for the same
    original example. This handles the case where validation examples have identical prompts
    but represent different rollouts.

    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts (used for logging but not grouping)
        infos_dict: variable name -> list of values for each sample
        n: Number of generations per original example
        seed: Random seed for bootstrap sampling

    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group samples by original example ID (every n samples belong to the same original example)
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        # Calculate original example ID: samples 0,1,2,3 -> example 0 (if n=4)
        original_example_id = sample_idx // n
        unique_id = f"example_{original_example_id}"
        var2vals = data_src2prompt2var2vals[data_source][unique_id]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group (rest of the logic remains the same)
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n_val = 2
                    while n_val < n_resps:
                        ns.append(n_val)
                        n_val *= 2
                    ns.append(n_resps)

                    for n_val in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n_val, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n_val}/mean"], metric[f"best@{n_val}/std"] = bon_mean, bon_std
                        metric[f"worst@{n_val}/mean"], metric[f"worst@{n_val}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n_val,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n_val}/mean"], metric[f"maj@{n_val}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
