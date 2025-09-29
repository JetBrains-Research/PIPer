import re
from typing import Literal


def remove_reasoning(answer: str) -> str:
    """
    Remove reasoning blocks from model responses by removing content before </think> tag.

    Args:
        answer: Response from the model that may contain reasoning blocks.

    Returns:
        Cleaned response with reasoning blocks removed.
    """

    if "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()
    return answer


def extract_bash_script(answer: str, mode: Literal['strict', 'lenient'] = 'strict') -> str:
    """
    Extract bash script from text using regex pattern ```bash```.
    Takes the last match if multiple are found.

    Args:
        answer: Response from the model.

    Returns:
        Extracted bash script or empty string if not found.
    """

    if mode == 'strict':
        pattern = r"```bash(.*?)```"
    elif mode == 'lenient':
        pattern = r"```bash(.*?)(?:```|$)"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    matches = list(re.finditer(pattern, answer, re.DOTALL))

    if not matches:
        return ""

    # Return the last match
    return matches[-1].group(1).strip()
