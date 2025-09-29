from langchain_core.tools import tool

from piper.reward_score.envsetup.shellcheck import get_readable_shellcheck_outputs


@tool
def shellcheck(script: str) -> str:
    """Run a syntax checker (shellcheck) on a given shell script."""

    return get_readable_shellcheck_outputs(script)
