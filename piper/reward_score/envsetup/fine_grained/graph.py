import argparse
import os
import re
import shlex
from collections import defaultdict
from typing import Callable, Dict, List, Literal, TypedDict, Union

from langchain_core.runnables.graph import CurveStyle, NodeStyles
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from piper.reward_score.envsetup.fine_grained.config_parsers import (
    check_poetry_all_extras_in_script,
    check_poetry_all_groups_in_script,
    extract_current_python_version_from_stdout,
    extract_executable_lines,
    extract_pip_extras_from_setup_cfg,
    extract_pip_extras_from_setup_py,
    extract_pip_extras_from_toml,
    extract_pip_python_version_from_setup_conf,
    extract_pip_used_extras_from_script,
    extract_poetry_groups_and_extras_from_toml,
    extract_poetry_python_version_from_conf,
    extract_poetry_used_extras_from_script,
    extract_poetry_used_groups_from_script,
    extract_pyenv_available_versions_from_stdout,
    extract_pyenv_install_versions_from_script,
    get_dependency_manager_from_pyproject_toml,
    validate_pip_install_commands,
    validate_poetry_install_commands,
)
from piper.reward_score.envsetup.fine_grained.constants import categorize_file


class PythonVersionReq(TypedDict):
    file: str
    version: str


class InstallCommandValidation(TypedDict):
    has_pip_install: bool
    has_poetry_install: bool
    pip_install_valid: bool
    poetry_install_valid: bool
    pip_validation_issues: List[str]
    poetry_validation_issues: List[str]


class FineGrainedContext(TypedDict, total=False):
    script: str
    repository: str
    true_dep_manager: Literal["poetry", "pip", "unclear"]
    used_dep_manager: Union[Literal["poetry", "pip", "unclear"], List[Literal["poetry", "pip", "unclear"]]]
    true_dep_groups: List[str]
    true_extras: List[str]
    used_dep_groups: List[str]
    used_extras: List[str]
    configuration_files: Dict[str, List[str]]
    python_version_reqs: List[PythonVersionReq]
    installed_python_versions: List[str]
    install_command_validation: InstallCommandValidation


def _extract_stdout(command_output: str) -> str:
    """Extract stdout content from execute_bash_command output."""
    if command_output.startswith("ERROR"):
        return ""

    if "stdout:\n" in command_output:
        stdout_section = command_output.split("stdout:\n")[1]
        if "\nstderr:" in stdout_section:
            stdout_section = stdout_section.split("\nstderr:")[0]
        return stdout_section.strip()

    return command_output.strip()


def create_fine_grained_graph(
    execute_bash_command: Callable,
    check_file_existence: bool = False,
) -> CompiledStateGraph:
    async def get_configuration_and_text_files(
        state: FineGrainedContext,
    ) -> FineGrainedContext:
        """Obtains the list of files in the repository
        and retains paths to configuration files based on the common patterns."""
        # Search all files, not just root directory
        output = await execute_bash_command(script="find . -type f", repository=state["repository"])
        stdout_content = _extract_stdout(output)

        files_list = stdout_content.splitlines()

        conf_files = defaultdict(list)

        for file in files_list:
            # Use the new pattern-based categorization
            category = categorize_file(file)
            if category:
                conf_files[category].append(file)

        return {"configuration_files": conf_files}

    async def refine_configuration_files(state: FineGrainedContext) -> FineGrainedContext:
        """Checks the contents of pyproject.toml files
        to determine whether the repository uses poetry or pip."""
        conf_files = state.get("configuration_files", {})
        unclear_files = conf_files.get("unclear", []).copy()  # Make a copy to avoid modification during iteration

        for file in unclear_files:
            if os.path.basename(file) != "pyproject.toml":
                continue

            output = await execute_bash_command(script=f"cat {shlex.quote(file)}", repository=state["repository"])
            file_contents = _extract_stdout(output)

            dm = get_dependency_manager_from_pyproject_toml(file_contents)

            if dm is not None:
                conf_files.get("unclear", []).remove(file)
                if dm not in conf_files:
                    conf_files[dm] = []
                conf_files[dm].append(file)

        if conf_files.get("poetry", []):
            return {"true_dep_manager": "poetry", "configuration_files": conf_files}

        if conf_files.get("pip", []):
            return {"true_dep_manager": "pip", "configuration_files": conf_files}

        return {"true_dep_manager": "unclear", "configuration_files": conf_files}

    async def get_python_version_requirements(
        state: FineGrainedContext,
    ) -> FineGrainedContext:
        """Checking the requirements for Python version in repository configuration files."""
        python_version_reqs: List[PythonVersionReq] = []

        for file in state.get("configuration_files", {}).get("poetry", []):
            if os.path.basename(file) == "pyproject.toml":
                output = await execute_bash_command(script=f"cat {shlex.quote(file)}", repository=state["repository"])
                file_contents = _extract_stdout(output)
                python_version = extract_poetry_python_version_from_conf(file_contents)
                if python_version is not None:
                    python_version_reqs.append({"file": file, "version": python_version})

        for file in state.get("configuration_files", {}).get("pip", []):
            if os.path.basename(file) == "pyproject.toml":
                output = await execute_bash_command(script=f"cat {shlex.quote(file)}", repository=state["repository"])
                file_contents = _extract_stdout(output)
                python_version = extract_pip_python_version_from_setup_conf(file_contents)
                if python_version is not None:
                    python_version_reqs.append({"file": file, "version": python_version})

        for file in state.get("configuration_files", {}).get("unclear", []):
            if os.path.basename(file) == ".python-version":
                output = await execute_bash_command(script=f"cat {shlex.quote(file)}", repository=state["repository"])
                file_contents = _extract_stdout(output)

                if file_contents:
                    python_version_reqs.append({"file": file, "version": file_contents})
        return {"python_version_reqs": python_version_reqs}

    async def get_installed_python_version(
        state: FineGrainedContext,
    ) -> FineGrainedContext:
        """Checking all the available Python versions after script execution."""
        script = state["script"]
        all_versions = []

        # Step 1: Extract versions being installed via pyenv in the script
        script_versions = extract_pyenv_install_versions_from_script(script)
        all_versions.extend(script_versions)

        # Step 2: Check which versions are available in pyenv
        pyenv_versions_result = await execute_bash_command(script="pyenv versions --bare", repository=state["repository"])
        pyenv_stdout = _extract_stdout(pyenv_versions_result)
        pyenv_versions = extract_pyenv_available_versions_from_stdout(pyenv_stdout)
        all_versions.extend(pyenv_versions)

        # Step 3: Check current active Python version
        current_python_result = await execute_bash_command(script="python --version", repository=state["repository"])
        current_stdout = _extract_stdout(current_python_result)
        current_version = extract_current_python_version_from_stdout(current_stdout)
        if current_version:
            all_versions.append(current_version)

        # Remove duplicates while preserving order
        unique_versions = set(all_versions)
        return {"installed_python_versions": list(unique_versions) if unique_versions else []}

    async def get_used_dep_manager(
        state: FineGrainedContext,
    ) -> FineGrainedContext:
        """Checking which dependency manager is used in the script."""
        script = state["script"]

        executable_lines = extract_executable_lines(script)
        executable_content = " ".join(executable_lines)

        has_poetry = bool(re.search(r"\bpoetry\s+install\b", executable_content))
        has_pip = bool(re.search(r"\bpip\s+install\b", executable_content) or re.search(r"\bpip3\s+install\b", executable_content))

        if has_poetry and has_pip:
            return {"used_dep_manager": ["poetry", "pip"]}
        if has_poetry:
            return {"used_dep_manager": "poetry"}
        if has_pip:
            return {"used_dep_manager": "pip"}
        return {"used_dep_manager": "unclear"}

    async def get_true_dep_groups_and_extras(
        state: FineGrainedContext,
    ) -> FineGrainedContext:
        """Parse dependency groups and extras defined in repository configuration files.

        Docs for Poetry:
        - https://python-poetry.org/docs/managing-dependencies/#optional-groups
        - https://python-poetry.org/docs/pyproject/#extras
        Docs for pip: https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies
        """
        true_dep_groups = []
        true_extras = []

        conf_files = state.get("configuration_files", {})

        # Check poetry configuration files
        for file in conf_files.get("poetry", []):
            if os.path.basename(file) == "pyproject.toml":
                output = await execute_bash_command(script=f"cat {shlex.quote(file)}", repository=state["repository"])
                file_contents = _extract_stdout(output)
                if file_contents:
                    groups, extras = extract_poetry_groups_and_extras_from_toml(file_contents)
                    true_dep_groups.extend(groups)
                    true_extras.extend(extras)

        # Check pip configuration files
        for file in conf_files.get("pip", []):
            output = await execute_bash_command(script=f"cat {shlex.quote(file)}", repository=state["repository"])
            file_contents = _extract_stdout(output)
            if file_contents:
                filename = os.path.basename(file)
                if filename == "pyproject.toml":
                    extras = extract_pip_extras_from_toml(file_contents)
                    true_extras.extend(extras)
                elif filename == "setup.py":
                    extras = extract_pip_extras_from_setup_py(file_contents)
                    true_extras.extend(extras)
                elif filename == "setup.cfg":
                    extras = extract_pip_extras_from_setup_cfg(file_contents)
                    true_extras.extend(extras)

        # Remove duplicates while preserving order
        true_dep_groups = list(dict.fromkeys(true_dep_groups))
        true_extras = list(dict.fromkeys(true_extras))

        return {"true_dep_groups": true_dep_groups, "true_extras": true_extras}

    async def get_used_dep_groups_and_extras(
        state: FineGrainedContext,
    ) -> FineGrainedContext:
        """Parse dependency groups and extras used in script installation commands.

        For Poetry, parses commands like:
        - poetry install --extras "mysql pgsql"
        - poetry install -E mysql -E pgsql
        - poetry install --all-extras
        - poetry install --with test,docs
        - poetry install --all-groups
        - poetry install --only test,docs
        """
        script = state["script"]
        used_dep_manager = state.get("used_dep_manager", "unclear")
        used_dep_groups = []
        used_extras = []

        # Handle Poetry commands
        if used_dep_manager == "poetry" or (isinstance(used_dep_manager, list) and "poetry" in used_dep_manager):
            # Extract specific groups and extras from install commands
            used_dep_groups.extend(extract_poetry_used_groups_from_script(script))
            used_extras.extend(extract_poetry_used_extras_from_script(script))

            # Handle --all-groups flag
            if check_poetry_all_groups_in_script(script):
                true_dep_groups = state.get("true_dep_groups", [])
                used_dep_groups.extend(true_dep_groups)

            # Handle --all-extras flag
            if check_poetry_all_extras_in_script(script):
                true_extras = state.get("true_extras", [])
                used_extras.extend(true_extras)

        # Handle pip commands
        if used_dep_manager == "pip" or (isinstance(used_dep_manager, list) and "pip" in used_dep_manager):
            # Extract extras from pip install commands
            used_extras.extend(extract_pip_used_extras_from_script(script))
            # Note: pip doesn't have dependency groups like Poetry

        # Remove duplicates while preserving order
        used_dep_groups = list(dict.fromkeys(used_dep_groups))
        used_extras = list(dict.fromkeys(used_extras))

        return {"used_dep_groups": used_dep_groups, "used_extras": used_extras}

    async def validate_install_commands(
        state: FineGrainedContext,
    ) -> FineGrainedContext:
        """Validates pip install and poetry install commands in the script.

        For pip install:
        - Validates syntax using documented pip install flags
        - For -r/--requirement flags, optionally checks file existence and content validity

        For poetry install:
        - Validates syntax using documented poetry install flags
        - Does NOT require --all-extras and --all-groups flags
        """
        script = state["script"]
        configuration_files = state.get("configuration_files", {})

        # Get all configuration files as a flat list
        all_config_files = []
        for files_list in configuration_files.values():
            all_config_files.extend(files_list)

        # Validate pip install commands with enhanced file validation
        has_pip_install, pip_install_valid, pip_validation_issues = await validate_pip_install_commands(script, all_config_files, check_file_existence=check_file_existence, execute_bash_command=execute_bash_command, repository=state["repository"])

        # Validate poetry install commands
        has_poetry_install, poetry_install_valid, poetry_validation_issues = validate_poetry_install_commands(script)

        return {
            "install_command_validation": {
                "has_pip_install": has_pip_install,
                "has_poetry_install": has_poetry_install,
                "pip_install_valid": pip_install_valid,
                "poetry_install_valid": poetry_install_valid,
                "pip_validation_issues": pip_validation_issues,
                "poetry_validation_issues": poetry_validation_issues,
            }
        }

    workflow = StateGraph(FineGrainedContext)
    workflow.add_node("get_configuration_and_text_files", get_configuration_and_text_files)
    workflow.add_node("refine_configuration_files", refine_configuration_files)
    workflow.add_node("get_python_version_requirements", get_python_version_requirements)
    workflow.add_node("get_installed_python_version", get_installed_python_version)
    workflow.add_node("get_used_dep_manager", get_used_dep_manager)
    workflow.add_node("get_true_dep_groups_and_extras", get_true_dep_groups_and_extras)
    workflow.add_node("get_used_dep_groups_and_extras", get_used_dep_groups_and_extras)
    workflow.add_node("validate_install_commands", validate_install_commands)

    workflow.add_edge(START, "get_configuration_and_text_files")
    workflow.add_edge("get_configuration_and_text_files", "refine_configuration_files")
    workflow.add_edge("refine_configuration_files", "get_true_dep_groups_and_extras")
    workflow.add_edge("get_true_dep_groups_and_extras", "get_used_dep_groups_and_extras")
    workflow.add_edge("get_used_dep_groups_and_extras", "get_python_version_requirements")
    workflow.add_edge("get_python_version_requirements", "validate_install_commands")
    workflow.add_edge("validate_install_commands", END)

    workflow.add_edge(START, "get_used_dep_manager")
    workflow.add_edge("get_used_dep_manager", END)

    workflow.add_edge(START, "get_installed_python_version")
    workflow.add_edge("get_installed_python_version", END)

    return workflow.compile()


def visualize_fine_grained_graph(output_dir: str) -> bool:
    graph = create_fine_grained_graph(execute_bash_command=None)
    try:
        os.makedirs(output_dir, exist_ok=True)
        png_file = os.path.join(output_dir, "fine_grained_graph.png")

        png_data = graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(
                first="#ffdfba",  # Light orange for start nodes
                last="#baffc9",  # Light green for end nodes
                default="#fad7de",  # Light pink for default nodes
            ),
            wrap_label_n_words=9,
            output_file_path=None,
            background_color="white",
            padding=10,
        )

        with open(png_file, "wb") as f:
            f.write(png_data)
        print(f"✓ Saved fine-grained graph image to {png_file}")
        return True

    except Exception as e:
        print(f"✗ Could not generate and save graph image: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the fine-grained graph and save it as PNG")
    parser.add_argument("--output-dir", "-o", type=str, default="img", help="Directory to save the generated graph image (default: img)")

    args = parser.parse_args()
    visualize_fine_grained_graph(output_dir=args.output_dir)
