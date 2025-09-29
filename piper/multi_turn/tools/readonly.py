import json
from typing import Annotated, Optional, TypedDict

import httpx
from envbench_graphs.readonly import EnvSetupReadOnlyState
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState


class EnvBenchExplorerResponse(TypedDict):
    stdout: str
    stderr: str
    exit_code: int
    success: bool


async def _send_bash_command_request(url: str, bash_command: str, repository: str, timeout: int = 30) -> EnvBenchExplorerResponse:
    """
    Sends a bash command execution request to the envbench-explorer API.

    Args:
        url: Server URL.
        bash_command: The bash command to execute
        repository: The repository context (e.g., "ansible/molecule")
        timeout: Request timeout in seconds (default: 30)

    Returns:
        Dict containing the API response with keys: stdout, stderr, exit_code, success
        If any error occurs (HTTP errors, network errors, JSON parsing), returns a dict
        with success=False and error details in stderr.

    Note:
        This function handles all exceptions internally and returns error information
        in the response dict rather than raising exceptions.
    """
    payload = {"bash_command": bash_command, "repository": repository}

    headers = {"Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, content=json.dumps(payload))

            # Raise an exception for bad status codes
            response.raise_for_status()

            # Parse and return the JSON response
            return response.json()

    except httpx.HTTPStatusError as e:
        # Handle HTTP status errors (4xx, 5xx)
        return {"stdout": "", "stderr": f"HTTP error {e.response.status_code}: {str(e)}", "exit_code": -1, "success": False}
    except httpx.RequestError as e:
        # Handle network/HTTP errors
        return {"stdout": "", "stderr": f"Request failed: {str(e)}", "exit_code": -1, "success": False}
    except json.JSONDecodeError as e:
        # Handle invalid JSON response
        return {"stdout": "", "stderr": f"Invalid JSON response: {str(e)}", "exit_code": -1, "success": False}
    except Exception as e:
        # Handle any other errors
        return {"stdout": "", "stderr": f"An unexpected error occurred: {str(e)}", "exit_code": -1, "success": False}


def _truncate_output(output: str, max_chars: Optional[int] = None) -> str:
    """
    Truncate output if it exceeds max_chars, keeping prefix and suffix.
    Tries to break at word/line boundaries when possible.

    Args:
        output: The output string to potentially truncate
        max_chars: Maximum number of characters to keep (None means no truncation)

    Returns:
        Truncated output with [X chars skipped] marker if truncation occurred
    """
    if max_chars is None:
        return output

    output_len = len(output)
    if output_len <= max_chars:
        return output

    # Calculate actual chars that will be skipped
    prefix_chars = max_chars // 2
    suffix_chars = max_chars - prefix_chars
    actual_skipped = output_len - prefix_chars - suffix_chars

    skip_marker = f"[{actual_skipped} chars skipped]"
    available_chars = max_chars - len(skip_marker)

    if available_chars <= 0:
        return skip_marker

    # Recalculate split with available space
    prefix_chars = available_chars // 2
    suffix_chars = available_chars - prefix_chars

    # Try to break at word/line boundaries
    def _find_break_point(text: str, target: int, from_end: bool = False) -> int:
        if target <= 0:
            return 0
        if target >= len(text):
            return len(text)

        # Look for line break first, then word break within reasonable distance
        search_range = min(50, target // 4)  # Search within 25% of target or 50 chars

        if from_end:
            start_pos = len(text) - target
            for i in range(max(0, start_pos - search_range), min(len(text), start_pos + search_range)):
                if text[i] == "\n":
                    return len(text) - i
                elif text[i] == " ":
                    return len(text) - i
            return target
        else:
            for i in range(max(0, target - search_range), min(len(text), target + search_range)):
                if text[i] == "\n":
                    return i
                elif text[i] == " ":
                    return i
            return target

    prefix_break = _find_break_point(output, prefix_chars)
    suffix_break = _find_break_point(output, suffix_chars, from_end=True)

    prefix = output[:prefix_break]
    suffix = output[-suffix_break:] if suffix_break > 0 else ""

    return prefix + skip_marker + suffix


async def send_bash_command_request(url: str, bash_command: str, repository: str, max_chars: Optional[int]) -> str:
    """
    Wrapper function that calls the envbench-explorer API and formats the response as a string.

    Args:
        url: Server URL.
        bash_command: The bash command to execute
        repository: The repository name.
        max_chars: Maximum number of characters to keep in total output (None means no truncation)

    Returns:
        Formatted string with command execution results in standardized format.
    """
    error_message = "ERROR: Could not execute given command."
    result = await _send_bash_command_request(url=url, bash_command=bash_command, repository=repository)

    # Issue #12: Calculate lengths once
    stdout_content = result.get("stdout", "")
    stderr_content = result.get("stderr", "")
    exit_code = result.get("exit_code", 0)

    # Issue #3: Skip processing if no truncation needed
    if max_chars is None:
        pass  # No truncation needed
    else:
        # Issue #4: Calculate actual header sizes dynamically
        base_format = "stdout:\n\n\nstderr:\n"
        error_prefix = error_message + "\n" if exit_code != 0 else ""

        # Issue #8: Account for error message in total limit
        reserved_chars = len(base_format) + len(error_prefix)
        available_chars = max(0, max_chars - reserved_chars)

        stdout_chars = len(stdout_content)
        stderr_chars = len(stderr_content)
        total_chars = stdout_chars + stderr_chars

        if total_chars > 0 and available_chars > 0:
            if stderr_chars == 0:
                stdout_max = available_chars
                stderr_max = None
            elif stdout_chars == 0:
                stdout_max = None
                stderr_max = available_chars
            else:
                # Split proportionally
                stdout_max = int(available_chars * stdout_chars / total_chars)
                stderr_max = available_chars - stdout_max

            stdout_content = _truncate_output(stdout_content, stdout_max)
            stderr_content = _truncate_output(stderr_content, stderr_max)

    # Issue #11: Build output in single operation
    parts = ["stdout:\n", stdout_content, "\n\nstderr:\n"]
    if stderr_content:
        parts.extend([stderr_content, "\n"])

    output = "".join(parts)

    # Issue #8: Handle error prefix
    if exit_code != 0:
        output = f"{error_message}\n{output}"

    # Issue #9: Final length validation and emergency truncation
    if max_chars is not None and len(output) > max_chars:
        # Emergency truncation if we somehow exceeded limit
        output = _truncate_output(output, max_chars)

    return output


def create_execute_bash_command_func(url: str, max_chars: Optional[int] = None):
    async def execute_bash_command(script: str, repository: str) -> str:
        """Execute a bash command and return the output."""
        response = await send_bash_command_request(url=url, bash_command=script, repository=repository, max_chars=max_chars)
        return response

    return execute_bash_command


def create_execute_bash_command_tool(url: str, max_chars: Optional[int] = None):
    @tool(parse_docstring=True)
    async def execute_bash_command(command: str, reason: str, state: Annotated[EnvSetupReadOnlyState, InjectedState]) -> str:
        """
        Executes a given bash command inside a Docker container in read-only mode.
        Only read operations are allowed - any write operations will fail.

        Args:
            command: A bash command with its arguments to be executed. Only READ operations are allowed.
            reason: A reason why you are calling the tool. For example, 'to check Python version' or 'to list directory contents'.
        """
        repository = state.get("tools_kwargs", {}).get("repository")
        if not repository:
            raise ValueError("Repository is not provided in the state.")
        response = await send_bash_command_request(url=url, bash_command=command, repository=repository, max_chars=max_chars)
        return response

    return execute_bash_command


@tool(parse_docstring=True)
def submit_shell_script(
    script: str,
) -> str:
    """
    Registers a shell script for environment setup for the current repository.
    This script can (and should!) contain write operations.

    Args:
        script: A shell script for environment setup.
    """
    return "Done! The script is submitted."
