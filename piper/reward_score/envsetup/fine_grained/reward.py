import asyncio
import json
from typing import Dict, List, Optional, TypedDict

from piper.multi_turn.tools.readonly import create_execute_bash_command_func
from piper.reward_score.envsetup.fine_grained.graph import FineGrainedContext, create_fine_grained_graph
from piper.reward_score.envsetup.utils import extract_bash_script, remove_reasoning

READONLY_SERVER_URL = "https://envbench-explorer.wlko.me/execute"
CHECK_FILE_EXISTENCE = True
MAX_CHARS = None


class FineGrainedRewardOutput(TypedDict):
    score: float
    format_errors: float
    failure_errors: float
    has_issues_errors: float
    # Specific error types
    multiple_dep_managers_error: float
    wrong_dep_manager_error: float
    invalid_install_syntax_error: float
    nonexistent_dep_groups_error: float
    nonexistent_extras_error: float
    incomplete_dep_groups_error: float
    incomplete_extras_error: float


def get_score_from_graph_output(graph_output: FineGrainedContext) -> FineGrainedRewardOutput:
    base_output = {
        "format_errors": 0.0,
        "failure_errors": 0.0,
        "has_issues_errors": 0.0,
        "multiple_dep_managers_error": 0.0,
        "wrong_dep_manager_error": 0.0,
        "invalid_install_syntax_error": 0.0,
        "nonexistent_dep_groups_error": 0.0,
        "nonexistent_extras_error": 0.0,
        "incomplete_dep_groups_error": 0.0,
        "incomplete_extras_error": 0.0,
    }

    # penalizing the hardest the patterns that most likely will result in a failure
    # 1. more than one dependency manager used
    used_dep_manager = graph_output.get("used_dep_manager")
    if isinstance(used_dep_manager, list) and len(used_dep_manager) == 1:
        used_dep_manager = used_dep_manager[0]
    if isinstance(used_dep_manager, list) and len(used_dep_manager) > 1:
        return {**base_output, "score": -0.5, "failure_errors": 1.0, "multiple_dep_managers_error": 1.0}

    # 2. using the wrong dependency manager
    true_dep_manager = graph_output.get("true_dep_manager")
    if true_dep_manager and used_dep_manager != true_dep_manager:
        return {**base_output, "score": -0.5, "failure_errors": 1.0, "wrong_dep_manager_error": 1.0}

    # 3. wrong syntax for pip install/poetry install
    # (including non-existent requirements files if the graph was configured accordingly)
    install_command_validation = graph_output.get("install_command_validation")
    if install_command_validation:
        if install_command_validation.get("has_pip_install", False) and not install_command_validation.get("pip_install_valid", True):
            return {**base_output, "score": -0.5, "failure_errors": 1.0, "invalid_install_syntax_error": 1.0}
        if install_command_validation.get("has_poetry_install", False) and not install_command_validation.get("poetry_install_valid", True):
            return {**base_output, "score": -0.5, "failure_errors": 1.0, "invalid_install_syntax_error": 1.0}

    # 4. installing non-existent dependency groups
    true_dep_groups = set(graph_output.get("true_dep_groups", []))
    used_dep_groups = set(graph_output.get("used_dep_groups", []))
    if used_dep_groups and used_dep_groups - true_dep_groups:
        return {**base_output, "score": -0.5, "failure_errors": 1.0, "nonexistent_dep_groups_error": 1.0}

    # 5. installing non-existent extras
    true_dep_extras = set(graph_output.get("true_extras", []))
    used_dep_extras = set(graph_output.get("used_extras", []))
    if used_dep_extras and used_dep_extras - true_dep_extras:
        return {**base_output, "score": -0.5, "failure_errors": 1.0, "nonexistent_extras_error": 1.0}

    # penalizing less the patterns that likely will not result in a failure
    # but still lead to non-ideal results
    # 1. not installing all the dependencies groups
    if true_dep_groups and used_dep_groups and used_dep_groups != true_dep_groups:
        return {**base_output, "score": 0.5, "has_issues_errors": 1.0, "incomplete_dep_groups_error": 1.0}

    # 2. not installing all the extras
    if true_dep_extras and used_dep_extras and used_dep_extras != true_dep_extras:
        return {**base_output, "score": 0.5, "has_issues_errors": 1.0, "incomplete_extras_error": 1.0}

    # TODO: consider Python installation/reqs

    return {**base_output, "score": 1.0}


async def fine_grained_reward_fn(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> FineGrainedRewardOutput:
    """
    Fine-grained reward that gathers context about the current repository and script.
    """
    solution_str = remove_reasoning(solution_str)

    script = extract_bash_script(solution_str)

    if not script:
        return {
            "score": -1.0,
            "format_errors": 1.0,
            "failure_errors": 0.0,
            "has_issues_errors": 0.0,
            "multiple_dep_managers_error": 0.0,
            "wrong_dep_manager_error": 0.0,
            "invalid_install_syntax_error": 0.0,
            "nonexistent_dep_groups_error": 0.0,
            "nonexistent_extras_error": 0.0,
            "incomplete_dep_groups_error": 0.0,
            "incomplete_extras_error": 0.0,
        }

    graph = create_fine_grained_graph(execute_bash_command=create_execute_bash_command_func(url=READONLY_SERVER_URL, max_chars=MAX_CHARS), check_file_existence=CHECK_FILE_EXISTENCE)
    graph_output = await graph.ainvoke({"script": script, "repository": extra_info.get("repository", "")})

    return get_score_from_graph_output(graph_output)


async def fine_grained_reward_fn_batched(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[dict]) -> List[FineGrainedRewardOutput]:
    """
    Batch version of fine_grained_reward_fn.
    """
    tasks = [fine_grained_reward_fn(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info) for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos)]
    return await asyncio.gather(*tasks)


async def messages_fine_grained_reward_fn(data_source: str, response: List[Dict[str, str]], ground_truth: str, extra_info: Optional[dict] = None) -> FineGrainedRewardOutput:
    """fine_grained_reward_fn for multi-turn setup that accepts raw conversation (list of messages)."""

    try:
        last_assistant_response = [msg for msg in response if msg["role"] == "assistant"][-1]

        if "tool_calls" in last_assistant_response:
            try:
                last_assistant_response = json.loads(last_assistant_response["tool_calls"][-1]["function"]["arguments"])["script"].strip()
                if not last_assistant_response.startswith("```bash"):
                    last_assistant_response = f"```bash\n{last_assistant_response}```"
            except (json.JSONDecodeError, KeyError):
                print(f"Encountered error while parsing tool call: {last_assistant_response}")
                return {
                    "score": -1.0,
                    "format_errors": 1.0,
                    "failure_errors": 0.0,
                    "has_issues_errors": 0.0,
                    "multiple_dep_managers_error": 0.0,
                    "wrong_dep_manager_error": 0.0,
                    "invalid_install_syntax_error": 0.0,
                    "nonexistent_dep_groups_error": 0.0,
                    "nonexistent_extras_error": 0.0,
                    "incomplete_dep_groups_error": 0.0,
                    "incomplete_extras_error": 0.0,
                }
        else:
            last_assistant_response = last_assistant_response["content"]
    except IndexError:
        return {
            "score": -1.0,
            "format_errors": 1.0,
            "failure_errors": 0.0,
            "has_issues_errors": 0.0,
            "multiple_dep_managers_error": 0.0,
            "wrong_dep_manager_error": 0.0,
            "invalid_install_syntax_error": 0.0,
            "nonexistent_dep_groups_error": 0.0,
            "nonexistent_extras_error": 0.0,
            "incomplete_dep_groups_error": 0.0,
            "incomplete_extras_error": 0.0,
        }

    return await fine_grained_reward_fn(data_source=data_source, solution_str=last_assistant_response, ground_truth=ground_truth, extra_info=extra_info)


async def messages_fine_grained_reward_fn_batched(data_sources: List[str], responses: List[List[Dict[str, str]]], ground_truths: List[str], extra_infos: List[dict]) -> List[FineGrainedRewardOutput]:
    """Batched version of messages_fine_grained_reward_fn."""

    tasks = [messages_fine_grained_reward_fn(data_source=data_source, response=response, ground_truth=ground_truth, extra_info=extra_info) for data_source, response, ground_truth, extra_info in zip(data_sources, responses, ground_truths, extra_infos)]
    return await asyncio.gather(*tasks)


# Sync wrapper functions
def fine_grained_reward_fn_sync(data_source: str, solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> FineGrainedRewardOutput:
    """Sync version of fine_grained_reward_fn."""
    return asyncio.run(fine_grained_reward_fn(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, extra_info=extra_info))


def fine_grained_reward_fn_batched_sync(data_sources: List[str], solution_strs: List[str], ground_truths: List[str], extra_infos: List[dict]) -> List[FineGrainedRewardOutput]:
    """Sync version of fine_grained_reward_fn_batched."""
    return asyncio.run(fine_grained_reward_fn_batched(data_sources=data_sources, solution_strs=solution_strs, ground_truths=ground_truths, extra_infos=extra_infos))


def messages_fine_grained_reward_fn_sync(data_source: str, response: List[Dict[str, str]], ground_truth: str, extra_info: Optional[dict] = None) -> FineGrainedRewardOutput:
    """Sync version of messages_fine_grained_reward_fn."""
    return asyncio.run(messages_fine_grained_reward_fn(data_source=data_source, response=response, ground_truth=ground_truth, extra_info=extra_info))


def messages_fine_grained_reward_fn_batched_sync(data_sources: List[str], responses: List[List[Dict[str, str]]], ground_truths: List[str], extra_infos: List[dict]) -> List[FineGrainedRewardOutput]:
    """Sync version of messages_fine_grained_reward_fn_batched."""
    return asyncio.run(messages_fine_grained_reward_fn_batched(data_sources=data_sources, responses=responses, ground_truths=ground_truths, extra_infos=extra_infos))
