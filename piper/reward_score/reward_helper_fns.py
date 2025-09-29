import asyncio
import re
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class RewardInput:
    batch_idx: int
    data_source: str
    solution_str: str
    ground_truth: str
    extra_info: dict


# Each output is either a float score or a dict containing a score key and some extra data
RewardOutput = float | dict

# A function that maps a RewardInput to a function that processes a list of RewardInputs and returns a list of RewardOutputs
ProcessorMapper = Callable[
    [RewardInput], Callable[[List[RewardInput]], List[RewardOutput]]
]


def create_reward_inputs(data_sources, solution_strs, ground_truths, extra_infos):
    return [
        RewardInput(idx, data_source, solution_str, ground_truth, extra_info)
        for idx, (data_source, solution_str, ground_truth, extra_info) in enumerate(
            zip(data_sources, solution_strs, ground_truths, extra_infos)
        )
    ]


def batch_process_with_mapper(
    items: List[RewardInput], processor_mapper: ProcessorMapper
) -> List[RewardOutput]:
    """
    A generic function to process a batch of items using a mapper function.

    Args:
        items: List of items to process
        processor_mapper: A function that maps each item to its appropriate processing function

    Returns:
        Results in the original order of the input items
    """
    # Group items by their processor function
    grouped_items = {}
    for idx, item in enumerate(items):
        processor = processor_mapper(item)
        # Use the function object's id as a key for grouping
        processor_id = id(processor)

        if processor_id not in grouped_items:
            grouped_items[processor_id] = {
                "processor": processor,
                "items": [],
                "indices": [],
            }

        grouped_items[processor_id]["items"].append(item)
        grouped_items[processor_id]["indices"].append(idx)

    # Process each group with its corresponding function
    results = [None] * len(items)  # Placeholder values

    for group_info in grouped_items.values():
        processor = group_info["processor"]
        batch_items = group_info["items"]
        indices = group_info["indices"]

        # Process this batch
        batch_results = processor(batch_items)

        # Place results in the correct positions
        for idx, result in zip(indices, batch_results):
            results[idx] = result

    return results


def add_prefix_to_dict(d: dict, prefix: str) -> dict:
    """
    Add a prefix to every key in a dictionary.

    Args:
        d: The input dictionary
        prefix: The prefix to add to each key

    Returns:
        A new dictionary with prefixed keys and original values
    """
    return {prefix + k: v for k, v in d.items()}


def strip_prompt_overlap_from_completion(prompt: str, completion: str) -> str:
    """
    Removes any overlapping suffix of the prompt that appears at the beginning of the completion.

    :param prompt: The original prompt string.
    :param completion: The raw model completion string.
    :return: Completion with the prompt suffix overlap stripped from the beginning.
    """
    max_overlap = 0
    for i in range(len(prompt)):
        suffix = prompt[i:]
        if completion.startswith(suffix):
            max_overlap = max(max_overlap, len(suffix))
    if max_overlap == 0:
        return '"No overlap"'
    else:
        return completion[max_overlap:]


def strip_first_line_from_code_block(code_block: str) -> str:
    """
    Strips the first line from the code block.
    """
    if "\n" not in code_block:
        return code_block
    return code_block.split("\n", 1)[1]


def extract_code_block(text: str, language: str = "kotlin") -> Optional[str]:
    """
    Extracts the Kotlin code block from a string, if present.

    The function looks for a code block delimited by triple backticks with
    the language identifier 'kotlin' and returns the enclosed code.

    :param text: The input string that may contain a Kotlin code block.
    :type text: str
    :return: The Kotlin code block content, or None if not found.
    :rtype: Optional[str]
    """
    match = re.search(rf"```{language}\s+(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_code_block_from_last_block(
    text: str, language: str = "kotlin"
) -> Optional[str]:
    """
    Extracts the last Kotlin/Python code block from a string, if present.

    The function looks for code blocks delimited by triple backticks with
    the language identifier 'kotlin' or 'python' and returns the enclosed code from the last block.

    :param text: The input string that may contain Kotlin code blocks.
    :type text: str
    :param language: The language identifier to look for (default: "kotlin")
    :type language: str
    :return: The last code block content, or None if not found.
    :rtype: Optional[str]
    """
    matches = re.findall(rf"```{language}\s+(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


import signal


class TimeoutError(Exception):
    pass


class timeout:

    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def my_getattr(obj, name):
    """getattr that works with hydra.utils.instantiate"""
    return getattr(obj, name)


def async_to_sync(fn):
    """convert an async function to a sync function"""
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


def sync_to_async(fn):
    """convert a sync function to an async function"""
    async def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper


def rubric(functions_with_weights):
    """
    Given a list of dicts, each with 'func' (callable) and optional 'weight' (float, default 1.0),
    returns a function that computes the weighted sum of the outputs of the functions,
    normalizing the weights to sum to 1.

    Example:
        r = rubric([
            {"func": lambda x: x, "weight": 2.0},
            {"func": lambda x: x**2, "weight": 1.0}
        ])
        r(3)  # returns (2/3)*3 + (1/3)*9 = 2 + 3 = 5

    Args:
        functions_with_weights (list[dict]): Each dict must have 'func' (callable), and optional 'weight' (float).

    Returns:
        Callable: A function that takes *args, **kwargs and returns the weighted sum.
    """
    # Prepare list of (func, weight)
    items = []
    for i, d in enumerate(functions_with_weights):
        func = d["func"]
        weight = d.get("weight", 1.0)
        name = d.get("name", f"func_{i}")
        items.append((func, weight, name))
    total_weight = sum(w for _, w, _ in items)
    if total_weight == 0:
        raise ValueError("Sum of weights must be nonzero.")
    norm_items = [(func, w / total_weight, name) for func, w, name in items]

    # TODO: in the future, we might want to run them in parallel
    def combined(data_sources, responses, ground_truths, extra_infos):
        outputs = [{'score': 0, 'reasoning': ""} for _ in responses]
        for i, (func, w, name) in enumerate(norm_items):
            outputs_i = func(data_sources, responses, ground_truths, extra_infos)
            for j, output in enumerate(outputs_i):
                outputs[j]['score'] += output['score'] * w
                outputs[j][name] = output['score']
                outputs[j]['reasoning'] += f"--- {name} ---\n"
                outputs[j]['reasoning'] += f"score: {output['score']} with weight {w}\n"
                for k, v in output.items():
                    if k == 'score':
                        continue
                    outputs[j]['reasoning'] += f"{k}: {v}\n"
                    # forward the extra info
                    outputs[j][f"{name}_{k}"] = v
        return outputs

    return combined
