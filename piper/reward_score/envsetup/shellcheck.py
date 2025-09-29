import json
import logging
import subprocess
from typing import Any, Dict, List, Optional

from piper.reward_score.envsetup.utils import extract_bash_script, remove_reasoning
from piper.reward_score.reward_helper_fns import RewardOutput

logger = logging.getLogger(__name__)


def get_readable_shellcheck_outputs(script: str) -> str:
    """Executes shellcheck on a given script and returns a readable string."""
    try:
        shellcheck_results = _run_shellcheck(script)
    except Exception as e:
        return f"Error running shellcheck: {repr(e)}"

    if shellcheck_results is None:
        return "Unable to obtain shellcheck results. Please, check the submitted script and try again."

    if len(shellcheck_results) == 0:
        return "No issues found."

    result_str = []
    result_str.append(f"Got {len(shellcheck_results)} issues reported by shellcheck. Please, check the submitted script and try again.")
    for i, result in enumerate(shellcheck_results):
        result_str.append(f"Issue {i + 1}")
        result_str.append(f"* Severity level: {result['level']}")
        result_str.append(f"* Message: '{result['message']}'")
        result_str.append(f"* Issue location: line {result['line']}")
    return "\n".join(result_str)


async def get_readable_shellcheck_outputs_async(script: str) -> str:
    """Async wrapper for get_readable_shellcheck_outputs."""
    return get_readable_shellcheck_outputs(script)


def _run_shellcheck(script: str) -> Optional[List[Dict[str, Any]]]:
    """
    Run shellcheck on a bash script and return results in JSON format.

    Args:
        script: Bash script to check.

    Returns:
        List of shellcheck results or None if an error occurred.
    """
    try:
        # Run shellcheck with JSON output format
        result = subprocess.run(
            ["shellcheck", "-s", "bash", "-f", "json", "-"],
            input=script,
            capture_output=True,
            text=True,
            check=False,
        )

        # Parse JSON output
        if result.stdout:
            # Parse each line as a separate JSON object
            return json.loads(result.stdout.strip())
        return []

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running shellcheck: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON output: {e}")
        return None


def _shellcheck_result_to_reward(shellcheck_results: Optional[List[Dict[str, Any]]]) -> float:
    """
    Convert shellcheck results to a reward score.

    Args:
        shellcheck_results: List of shellcheck results.

    Returns:
        float: Reward score between 0.0 and 1.0.
    """
    if shellcheck_results is None:
        return 0.0
    if len(shellcheck_results) == 0:
        return 1.0
    issues_count = len(shellcheck_results)
    return max(1.0 - issues_count / 20, 0.0)


def shellcheck_reward_fn(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> RewardOutput:
    r"""Reward function that evaluates bash scripts using shellcheck.

    Extracts bash scripts from responses and runs shellcheck on them.
    Returns rewards based on shellcheck results:
    - -1.0 if no bash script is found or shellcheck fails
    - 1.0 if no issues are found
    - Decreasing score based on number of issues (1.0 - issues_count/20)

    Args:
        data_source (str): Source of the data.
        solution_str (str): Solution string containing the bash script.
        ground_truth (str): Ground truth string.
        extra_info (Optional[dict], optional): Additional information. Defaults to None.

    Returns:
        RewardOutput: Dictionary containing the score and number of shebang errors.
    """
    solution_str = remove_reasoning(solution_str)

    script = extract_bash_script(solution_str)

    if not script:
        return {"score": -1.0, "num_shebang_errors": 0.0}

    shellcheck_results = _run_shellcheck(script)

    # shellcheck error codes: https://gist.github.com/nicerobot/53cee11ee0abbdc997661e65b348f375
    return {"score": _shellcheck_result_to_reward(shellcheck_results), "num_shebang_errors": len([result for result in shellcheck_results if result["code"] in [1128, 2148]]) if shellcheck_results else 0.0}


def shellcheck_reward_fn_batched(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[dict]) -> List[RewardOutput]:
    """
    Batch version of shellcheck_reward_fn.
    """
    return [shellcheck_reward_fn(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info) for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos)]


def messages_shellcheck_reward_fn(data_source: str, response: List[Dict[str, str]], ground_truth: str, extra_info: Optional[dict] = None) -> RewardOutput:
    r"""shellcheck_reward_fn for multi-turn setup that accepts raw conversation (list of messages)."""

    try:
        last_assistant_response = [msg for msg in response if msg["role"] == "assistant"][-1]

        if "tool_calls" in last_assistant_response:
            try:
                last_assistant_response = json.loads(last_assistant_response["tool_calls"][-1]["function"]["arguments"])["script"].strip()
                if not last_assistant_response.startswith("```bash"):
                    last_assistant_response = f"```bash\n{last_assistant_response}```"
            except (json.JSONDecodeError, KeyError):
                print(f"Encountered error while parsing tool call: {last_assistant_response}")
                return {"score": -1.0, "num_shebang_errors": 0.0}
        else:
            last_assistant_response = last_assistant_response["content"]
    except IndexError:
        return {"score": -1.0, "num_shebang_errors": 0.0}

    return shellcheck_reward_fn(data_source=data_source, solution_str=last_assistant_response, ground_truth=ground_truth, extra_info=extra_info)


def messages_shellcheck_reward_fn_batched(data_sources: List[str], responses: List[List[Dict[str, str]]], ground_truths: List[str], extra_infos: List[dict]) -> List[RewardOutput]:
    r"""Batched version of messages_shellcheck_reward_fn."""

    return [messages_shellcheck_reward_fn(data_source=data_source, response=response, ground_truth=ground_truth, extra_info=extra_info) for data_source, response, ground_truth, extra_info in zip(data_sources, responses, ground_truths, extra_infos)]


def strict_messages_shellcheck_reward_fn(data_source: str, response: List[Dict[str, str]], ground_truth: str, extra_info: Optional[dict] = None) -> RewardOutput:
    """
    Strict version of messages_shellcheck_reward_fn.
    Returns 1.0 if there are no shellcheck errors, 0.0 if there are errors, and -1.0 if the format fails.
    """
    try:
        last_assistant_response = [msg for msg in response if msg["role"] == "assistant"][-1]

        if "tool_calls" in last_assistant_response:
            try:
                last_assistant_response: str = json.loads(last_assistant_response["tool_calls"][-1]["function"]["arguments"])["script"].strip()
                if not last_assistant_response.startswith("```bash"):
                    last_assistant_response = f"```bash\n{last_assistant_response}```"
            except (json.JSONDecodeError, KeyError):
                print(f"Encountered error while parsing tool call: {last_assistant_response}")
                return {"score": -1.0, "num_shebang_errors": 0.0}
        else:
            last_assistant_response = last_assistant_response["content"]
    except IndexError:
        return {"score": -1.0, "num_shebang_errors": 0.0}

    solution_str = remove_reasoning(last_assistant_response)
    script = extract_bash_script(solution_str)

    if not script:
        return {"score": -1.0, "num_shebang_errors": 0.0}

    shellcheck_results = _run_shellcheck(script)
    num_errors = len(shellcheck_results) if shellcheck_results else 0

    score = 1.0 if num_errors == 0 else 0.0
    return {"score": score, "num_shebang_errors": len([result for result in shellcheck_results if result["code"] in [1128, 2148]]) if shellcheck_results else 0.0}


def strict_messages_shellcheck_reward_fn_batched(data_sources: List[str], responses: List[List[Dict[str, str]]], ground_truths: List[str], extra_infos: List[dict]) -> List[RewardOutput]:
    """
    Batched version of strict_messages_shellcheck_reward_fn.
    """
    return [strict_messages_shellcheck_reward_fn(data_source=data_source, response=response, ground_truth=ground_truth, extra_info=extra_info) for data_source, response, ground_truth, extra_info in zip(data_sources, responses, ground_truths, extra_infos)]
