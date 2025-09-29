import ast
import configparser
import re
import shlex
from typing import Callable, List, Optional, Tuple

try:
    import bashlex
except ImportError:
    bashlex = None

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

try:
    from packaging.requirements import InvalidRequirement, Requirement
except ImportError:
    # Fallback for systems without packaging library
    Requirement = None
    InvalidRequirement = Exception


def extract_poetry_groups_and_extras_from_toml(file_contents: str) -> Tuple[List[str], List[str]]:
    """Extract poetry dependency groups and extras from pyproject.toml.

    Dependency groups: [tool.poetry.group.{name}.dependencies]
    Extras: [tool.poetry.extras]
    See: https://python-poetry.org/docs/managing-dependencies/#optional-groups
    """
    try:
        toml_contents = tomllib.loads(file_contents)
        groups = []
        extras = []

        tool_poetry = toml_contents.get("tool", {}).get("poetry", {})

        # Check for dependency groups: [tool.poetry.group.{name}.dependencies]
        # Only extract groups that are marked as optional = true
        poetry_groups = tool_poetry.get("group", {})
        for group_name in poetry_groups.keys():
            group_data = poetry_groups[group_name]
            if isinstance(group_data, dict):
                # Only include groups that are explicitly marked as optional
                is_optional = group_data.get("optional", False)
                has_dependencies = "dependencies" in group_data or "dev-dependencies" in group_data
                if is_optional and has_dependencies:
                    groups.append(group_name)

        # Check for extras: [tool.poetry.extras]
        poetry_extras = tool_poetry.get("extras", {})
        extras.extend(poetry_extras.keys())

        return groups, extras
    except tomllib.TOMLDecodeError:
        return [], []


def extract_poetry_python_version_from_conf(file_contents: str) -> Optional[str]:
    """Extract Python version requirement from poetry configuration.

    See: https://python-poetry.org/docs/basic-usage/#setting-a-python-version
    """
    try:
        toml_contents = tomllib.loads(file_contents)
        return toml_contents.get("tool", {}).get("poetry", {}).get("dependencies", {}).get("python")
    except tomllib.TOMLDecodeError:
        return None


def extract_pip_extras_from_setup_py(file_contents: str) -> List[str]:
    """Extract pip extras from setup.py using AST parsing.

    Handles various patterns:
    - setup(extras_require={"dev": [...], "test": [...]})
    - extras = {"dev": [...]}; setup(extras_require=extras)
    - Direct dictionary definitions and variable references

    See: https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies
    """
    extras = []

    try:
        # Parse the Python file into an AST
        tree = ast.parse(file_contents)

        # Dictionary to store variable assignments
        variables = {}

        class SetupVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                """Capture variable assignments."""
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        # Try to evaluate simple expressions
                        try:
                            if isinstance(node.value, ast.Dict):
                                # Handle dictionary assignments
                                dict_value = {}
                                for k, v in zip(node.value.keys, node.value.values):
                                    if isinstance(k, ast.Constant):
                                        dict_value[k.value] = v
                                variables[var_name] = dict_value
                            elif isinstance(node.value, ast.List):
                                # Handle list assignments
                                list_value = []
                                for item in node.value.elts:
                                    if isinstance(item, ast.Constant):
                                        list_value.append(item.value)
                                variables[var_name] = list_value
                        except Exception:
                            pass
                self.generic_visit(node)

            def visit_Call(self, node):
                """Look for setup() calls."""
                if isinstance(node.func, ast.Name) and node.func.id == "setup":
                    # Look for extras_require keyword argument
                    for keyword in node.keywords:
                        if keyword.arg == "extras_require":
                            extras_value = self._extract_extras_from_node(keyword.value)
                            if extras_value:
                                extras.extend(extras_value)
                self.generic_visit(node)

            def _extract_extras_from_node(self, node):
                """Extract extras keys from an AST node."""
                node_extras = []

                if isinstance(node, ast.Dict):
                    # Direct dictionary: {"dev": [...], "test": [...]}
                    for key in node.keys:
                        if isinstance(key, ast.Constant):
                            node_extras.append(key.value)
                        elif isinstance(key, ast.Str):  # Python < 3.8 compatibility
                            node_extras.append(key.s)

                elif isinstance(node, ast.Name):
                    # Variable reference: extras_require=my_extras
                    var_name = node.id
                    if var_name in variables:
                        var_value = variables[var_name]
                        if isinstance(var_value, dict):
                            node_extras.extend(var_value.keys())
                        elif isinstance(var_value, list):
                            # Sometimes extras_require is a list of extra names
                            node_extras.extend(var_value)

                return node_extras

        # Visit all nodes in the AST
        visitor = SetupVisitor()
        visitor.visit(tree)

    except (SyntaxError, ValueError):
        # Fall back to regex parsing if AST parsing fails
        extras_pattern = r"extras_require\s*=\s*{([^}]+)}"
        match = re.search(extras_pattern, file_contents, re.DOTALL)
        if match:
            extras_content = match.group(1)
            key_pattern = r'["\']([^"\']+)["\']\s*:'
            extras.extend(re.findall(key_pattern, extras_content))

    return extras


def extract_pip_extras_from_setup_cfg(file_contents: str) -> List[str]:
    """Extract pip extras from setup.cfg.

    Looks for [options.extras_require] section.
    See: https://setuptools.pypa.io/en/latest/userguide/declarative_config.html#using-a-src-layout
    """
    extras = []
    try:
        config = configparser.ConfigParser()
        config.read_string(file_contents)

        # Check for [options.extras_require] section
        if config.has_section("options.extras_require"):
            extras.extend(config.options("options.extras_require"))

    except (configparser.Error, Exception):
        # Fall back to regex parsing if configparser fails
        lines = file_contents.split("\n")
        in_extras_section = False

        for line in lines:
            line = line.strip()
            if line == "[options.extras_require]":
                in_extras_section = True
                continue
            elif line.startswith("[") and line.endswith("]"):
                in_extras_section = False
                continue

            if in_extras_section and "=" in line:
                key = line.split("=")[0].strip()
                if key:
                    extras.append(key)

    return extras


def extract_pip_extras_from_toml(file_contents: str) -> List[str]:
    """Extract extras when pyproject.toml + pip are used.

    See https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies.
    """
    try:
        toml_contents = tomllib.loads(file_contents)
        # Check for [project.optional-dependencies]
        optional_deps = toml_contents.get("project", {}).get("optional-dependencies", {})
        return list(optional_deps.keys())
    except tomllib.TOMLDecodeError:
        return []


def extract_pip_python_version_from_setup_conf(file_contents: str) -> Optional[str]:
    """Extract Python version requirement from pip setup configuration.

    See: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#python-requires
    """
    try:
        toml_contents = tomllib.loads(file_contents)
        return toml_contents.get("project", {}).get("requires-python")
    except tomllib.TOMLDecodeError:
        return None


def get_dependency_manager_from_pyproject_toml(file_contents: str) -> Optional[str]:
    """Determine dependency manager from pyproject.toml build-system.

    See: https://packaging.python.org/en/latest/guides/writing-pyproject-toml
    """
    from piper.reward_score.envsetup.fine_grained.constants import BUILD_BACKEND_TO_DM_MAP

    try:
        toml_contents = tomllib.loads(file_contents)
        build_system = toml_contents.get("build-system")
        if build_system is None:
            return None
        build_backend = build_system.get("build-backend")
        if build_backend in BUILD_BACKEND_TO_DM_MAP:
            return BUILD_BACKEND_TO_DM_MAP[build_backend]
        return None
    except tomllib.TOMLDecodeError as e:
        print(f"get_dependency_manager_from_pyproject_toml: Error parsing TOML: {e}")
        return None


def extract_executable_lines(script_content: str) -> List[str]:
    """Extract lines that are actually executable (not comments)."""
    lines = script_content.split("\n")
    executable_lines = []

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue
        if "#" in stripped_line:
            code_part = stripped_line.split("#")[0].strip()
            if code_part:
                executable_lines.append(code_part)
        else:
            executable_lines.append(stripped_line)

    return executable_lines


def find_install_commands_with_bashlex(script_content: str, limit_depth: Optional[int] = None) -> List[dict]:
    """
    Use bashlex to find pip install and poetry install commands with proper context.

    Returns a list of dictionaries with:
    - command_type: 'pip' or 'poetry'
    - command_text: the full command text
    - arguments: list of arguments
    - has_variables: whether the command contains variables
    - in_conditional: whether the command is in a conditional context
    - requirement_files: list of requirement files referenced
    """
    if bashlex is None:
        raise ImportError("bashlex is required for robust bash parsing. Install with: pip install bashlex")

    tree = bashlex.parse(script_content)

    install_commands = []
    visited = set()

    def traverse_node(node, in_conditional=False, depth=0):
        """Recursively traverse the AST to find install commands."""
        # Prevent infinite recursion with depth limit and visited tracking
        if (limit_depth is not None and depth > limit_depth) or id(node) in visited:
            return
        visited.add(id(node))

        # Check if we're entering a conditional context
        if node.kind in ["if", "while", "for", "case"]:
            in_conditional = True

        if node.kind == "command":
            # Extract the command parts
            parts = []
            command_text = script_content[node.pos[0] : node.pos[1]]

            for part in node.parts:
                if part.kind == "word":
                    word_text = script_content[part.pos[0] : part.pos[1]]
                    parts.append(word_text)

            if parts:
                cmd = parts[0]
                args = parts[1:]

                # Check for pip install commands
                if cmd in ["pip", "pip3"] and len(args) > 0 and args[0] == "install":
                    install_commands.append(
                        {
                            "command_type": "pip",
                            "command_text": command_text,
                            "arguments": args[1:],  # Skip 'install'
                            "has_variables": _contains_variables(node, script_content, limit_depth=limit_depth),
                            "in_conditional": in_conditional,
                            "requirement_files": _extract_requirement_files(args[1:]),
                        }
                    )

                # Check for python -m pip install commands
                elif cmd in ["python", "python3"] and len(args) >= 3 and args[0] == "-m" and args[1] == "pip" and args[2] == "install":
                    install_commands.append(
                        {
                            "command_type": "pip",
                            "command_text": command_text,
                            "arguments": args[3:],  # Skip '-m pip install'
                            "has_variables": _contains_variables(node, script_content, limit_depth=limit_depth),
                            "in_conditional": in_conditional,
                            "requirement_files": _extract_requirement_files(args[3:]),
                        }
                    )

                # Check for poetry install commands
                elif cmd == "poetry" and len(args) > 0 and args[0] == "install":
                    install_commands.append(
                        {
                            "command_type": "poetry",
                            "command_text": command_text,
                            "arguments": args[1:],  # Skip 'install'
                            "has_variables": _contains_variables(node, script_content, limit_depth=limit_depth),
                            "in_conditional": in_conditional,
                            "requirement_files": [],  # Poetry doesn't use requirement files
                        }
                    )

        # Recursively traverse child nodes (both list and parts)
        if hasattr(node, "list") and node.list:
            for child in node.list:
                traverse_node(child, in_conditional, depth + 1)
        if hasattr(node, "parts") and node.parts:
            for part in node.parts:
                traverse_node(part, in_conditional, depth + 1)

    for node in tree:
        traverse_node(node)

    return install_commands


def _contains_variables(node, script_content: str, limit_depth: Optional[int] = None) -> bool:
    """Check if a command node contains parameter expansions (variables)."""
    if bashlex is None:
        return False

    # Look for parameter expansion nodes in the AST with recursion safeguards
    visited = set()

    def check_node(n, depth=0):
        # Prevent infinite recursion with depth limit and visited tracking
        if (limit_depth is not None and depth > limit_depth) or id(n) in visited:
            return False
        visited.add(id(n))

        if n.kind == "parameter":
            return True
        if hasattr(n, "list") and n.list:
            for child in n.list:
                if check_node(child, depth + 1):
                    return True
        if hasattr(n, "parts") and n.parts:
            for part in n.parts:
                if check_node(part, depth + 1):
                    return True
        return False

    return check_node(node)


def _extract_requirement_files(args: List[str]) -> List[str]:
    """Extract requirement file paths from pip install arguments."""
    req_files = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ["-r", "--requirement"]:
            if i + 1 < len(args):
                req_files.append(args[i + 1])
                i += 2
            else:
                i += 1
        elif arg.startswith("--requirement="):
            req_files.append(arg.split("=", 1)[1])
            i += 1
        else:
            i += 1
    return req_files


def should_skip_file_validation_bashlex(command_info: dict) -> bool:
    """
    Determine if file validation should be skipped based on bashlex analysis.

    Skip validation if:
    - Command contains variables
    - Command is in a conditional context
    """
    return command_info["has_variables"] or command_info["in_conditional"]


def _contains_variable_patterns(text: str) -> bool:
    """Check if text contains bash variable patterns or expressions."""
    # Check for common variable patterns
    variable_patterns = [
        r"\$\w+",  # $VAR
        r"\$\{[^}]+\}",  # ${VAR}
        r"\$\([^)]+\)",  # $(cmd)
        r"`[^`]+`",  # `cmd`
        r"~",  # home directory expansion
    ]

    for pattern in variable_patterns:
        if re.search(pattern, text):
            return True
    return False


# Functions needed by other parts of the codebase (kept minimal)
def extract_pyenv_install_versions_from_script(script_content: str) -> List[str]:
    """Extract Python versions being installed via pyenv in the script."""
    if bashlex is None:
        raise ImportError("bashlex is required for robust bash parsing. Install with: pip install bashlex")

    try:
        tree = bashlex.parse(script_content)
    except Exception:
        # Handle any bashlex parsing errors (syntax errors, etc.)
        return []

    versions = []
    visited = set()

    def traverse_node(node, depth=0, limit_depth: Optional[int] = None):
        """Recursively traverse the AST to find pyenv install commands."""
        # Prevent infinite recursion with depth limit and visited tracking
        if (limit_depth is not None and depth > limit_depth) or id(node) in visited:
            return
        visited.add(id(node))

        if node.kind == "command":
            # Extract the command parts
            parts = []
            for part in node.parts:
                if part.kind == "word":
                    word_text = script_content[part.pos[0] : part.pos[1]]
                    parts.append(word_text)

            if len(parts) >= 3 and parts[0] == "pyenv" and parts[1] == "install":
                # Find the first non-flag argument after 'pyenv install'
                for i in range(2, len(parts)):
                    arg = parts[i]
                    # Skip flags (anything starting with -)
                    if not arg.startswith("-"):
                        # Remove quotes if present
                        version = arg.strip("'\"")
                        # Only include literal string versions, not variables or expressions
                        if not _contains_variable_patterns(version):
                            versions.append(version)
                        break

        # Recursively traverse child nodes (both list and parts)
        if hasattr(node, "list") and node.list:
            for child in node.list:
                traverse_node(child, depth + 1)
        if hasattr(node, "parts") and node.parts:
            for part in node.parts:
                traverse_node(part, depth + 1)

    for node in tree:
        traverse_node(node)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(versions))


def extract_current_python_version_from_stdout(stdout_content: str) -> Optional[str]:
    """Extract Python version from python --version stdout content."""
    if not stdout_content:
        return None

    # Extract just the version number from "Python X.Y.Z"
    version_match = re.search(r"Python\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)", stdout_content)
    if version_match:
        return version_match.group(1)

    return None


def extract_pyenv_available_versions_from_stdout(stdout_content: str) -> List[str]:
    """Extract available Python versions from pyenv versions stdout content."""
    if not stdout_content:
        return []

    pyenv_available = stdout_content.split("\n")
    versions = []
    for version in pyenv_available:
        if version.strip():  # Skip empty lines
            versions.append(version.strip())

    return versions


def extract_poetry_used_groups_from_script(script_content: str, limit_depth: Optional[int] = None) -> List[str]:
    """Extract dependency groups used in poetry install commands from script content."""
    if bashlex is None:
        raise ImportError("bashlex is required for robust bash parsing. Install with: pip install bashlex")

    try:
        commands = find_install_commands_with_bashlex(script_content, limit_depth=limit_depth)
        poetry_commands = [cmd for cmd in commands if cmd["command_type"] == "poetry"]

        used_groups = []
        for cmd in poetry_commands:
            args = cmd["arguments"]
            i = 0
            while i < len(args):
                if args[i] == "--with" and i + 1 < len(args):
                    groups = args[i + 1].split(",")
                    used_groups.extend(groups)
                    i += 2
                elif args[i] == "--only" and i + 1 < len(args):
                    groups = args[i + 1].split(",")
                    used_groups.extend(groups)
                    i += 2
                else:
                    i += 1

        return list(dict.fromkeys(used_groups))  # Remove duplicates
    except Exception:
        return []


def extract_poetry_used_extras_from_script(script_content: str, limit_depth: Optional[int] = None) -> List[str]:
    """Extract extras used in poetry install commands from script content."""
    if bashlex is None:
        raise ImportError("bashlex is required for robust bash parsing. Install with: pip install bashlex")

    try:
        commands = find_install_commands_with_bashlex(script_content, limit_depth=limit_depth)
        poetry_commands = [cmd for cmd in commands if cmd["command_type"] == "poetry"]

        used_extras = []
        for cmd in poetry_commands:
            args = cmd["arguments"]
            i = 0
            while i < len(args):
                if args[i] == "--extras" and i + 1 < len(args):
                    extras = args[i + 1].strip("\"'").split()
                    used_extras.extend(extras)
                    i += 2
                elif args[i] == "-E" and i + 1 < len(args):
                    used_extras.append(args[i + 1])
                    i += 2
                else:
                    i += 1

        return list(dict.fromkeys(used_extras))  # Remove duplicates
    except Exception:
        return []


def check_poetry_all_groups_in_script(script_content: str, limit_depth: Optional[int] = None) -> bool:
    """Check if --all-groups flag is used in poetry install commands."""
    if bashlex is None:
        raise ImportError("bashlex is required for robust bash parsing. Install with: pip install bashlex")

    try:
        commands = find_install_commands_with_bashlex(script_content, limit_depth=limit_depth)
        poetry_commands = [cmd for cmd in commands if cmd["command_type"] == "poetry"]

        for cmd in poetry_commands:
            if "--all-groups" in cmd["arguments"]:
                return True
        return False
    except Exception:
        return False


def check_poetry_all_extras_in_script(script_content: str, limit_depth: Optional[int] = None) -> bool:
    """Check if --all-extras flag is used in poetry install commands."""
    if bashlex is None:
        raise ImportError("bashlex is required for robust bash parsing. Install with: pip install bashlex")

    try:
        commands = find_install_commands_with_bashlex(script_content, limit_depth=limit_depth)
        poetry_commands = [cmd for cmd in commands if cmd["command_type"] == "poetry"]

        for cmd in poetry_commands:
            if "--all-extras" in cmd["arguments"]:
                return True
        return False
    except Exception:
        return False


def extract_pip_used_extras_from_script(script_content: str, limit_depth: Optional[int] = None) -> List[str]:
    """Extract extras used in pip install commands from script content.

    Only collects extras for the current package (indicated by '.' or '-e .'),
    not for external packages.
    """
    if bashlex is None:
        raise ImportError("bashlex is required for robust bash parsing. Install with: pip install bashlex")

    try:
        commands = find_install_commands_with_bashlex(script_content, limit_depth=limit_depth)
        pip_commands = [cmd for cmd in commands if cmd["command_type"] == "pip"]

        used_extras = []
        for cmd in pip_commands:
            args = cmd["arguments"]
            is_editable = False

            # Check if this is an editable install
            if "-e" in args or "--editable" in args:
                is_editable = True

            for arg in args:
                # Skip flags
                if arg.startswith("-"):
                    continue

                # Check if this argument has extras
                if "[" in arg and "]" in arg:
                    # Check if it's a current package pattern
                    is_current_package = False

                    # Pattern 1: starts with '.' (current directory)
                    if arg.startswith("."):
                        is_current_package = True
                    # Pattern 2: editable install (any path when -e is used)
                    elif is_editable:
                        is_current_package = True
                    # Pattern 3: local path (starts with ./ or ../ or /)
                    elif arg.startswith("./") or arg.startswith("../") or arg.startswith("/"):
                        is_current_package = True

                    if is_current_package:
                        match = re.search(r"\[([^\]]+)\]", arg)
                        if match:
                            extras = [e.strip() for e in match.group(1).split(",")]
                            used_extras.extend(extras)

        return list(dict.fromkeys(used_extras))  # Remove duplicates
    except Exception:
        return []


def validate_requirements_file_content(content: str) -> Tuple[bool, List[str]]:
    """Validate if content appears to be a valid pip requirements file.

    Uses the packaging library's PEP 508 implementation for proper validation.

    Args:
        content: The file content to validate

    Returns:
        Tuple of (is_valid, validation_issues)
    """
    if not content.strip():
        return True, []  # Empty file is valid

    if Requirement is None:
        # Fallback to basic validation if packaging library is not available
        print("Warning: packaging library is not available, using fallback validation")
        return _validate_requirements_file_content_fallback(content)

    issues = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Skip pip options like -f, --find-links, etc.
        if line.startswith("-"):
            continue

        # Use packaging library to validate the requirement according to PEP 508
        try:
            Requirement(line)
        except InvalidRequirement as e:
            issues.append(f"Line {line_num}: Invalid requirement '{line}': {str(e)}")
        except Exception as e:
            # Catch any other parsing errors
            issues.append(f"Line {line_num}: Error parsing requirement '{line}': {str(e)}")

    return len(issues) == 0, issues


def _validate_requirements_file_content_fallback(content: str) -> Tuple[bool, List[str]]:
    """Fallback validation when packaging library is not available.

    This uses simplified regex-based validation as a backup.
    """
    issues = []
    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        # Skip pip options like -f, --find-links, etc.
        if line.startswith("-"):
            continue

        # Basic validation for package specifications
        if not _is_valid_requirement_line_fallback(line):
            issues.append(f"Line {line_num}: Invalid requirement format: {line}")

    return len(issues) == 0, issues


def _is_valid_requirement_line_fallback(line: str) -> bool:
    """Fallback requirement line validation using regex.

    This is a simplified validation that checks for common patterns.
    Only used when the packaging library is not available.
    """
    # Very basic patterns for valid requirements
    valid_patterns = [
        r"^[a-zA-Z0-9\-_\.]+(\[[a-zA-Z0-9\-_,\s]+\])?([<>=!~]+[a-zA-Z0-9\-_\.\*]+(\.[a-zA-Z0-9\-_\*]+)*)?$",  # package[extras]==1.0.0
        r"^-e\s+.+$",  # editable installs
        r"^git\+.+$",  # git URLs
        r"^https?://.+$",  # HTTP URLs
        r"^file://.+$",  # file URLs
        r"^[a-zA-Z0-9\-_\.]+\s*$",  # simple package names
    ]

    for pattern in valid_patterns:
        if re.match(pattern, line):
            return True

    # Check for common invalid content that clearly isn't a requirements file
    if any(keyword in line.lower() for keyword in ["<html>", "<!doctype", "error:", "not found"]):
        return False

    # If line contains clearly invalid characters for package names, reject it
    if any(char in line for char in ["!", "@", "$", "%", "&", "(", ")", "{", "}", "|", "\\", "`", "~"]):
        return False

    return True


async def validate_pip_install_commands(script_content: str, available_files: List[str], check_file_existence: bool = False, execute_bash_command: Optional[Callable] = None, repository: Optional[str] = None) -> Tuple[bool, bool, List[str]]:
    """Validate pip install commands in script content.

    Args:
        script_content: The bash script content to analyze
        available_files: List of available configuration files to check -r references against
        check_file_existence: Whether to check if referenced files exist in available_files (default: False)
        execute_bash_command: Optional function to execute bash commands for file content validation
        repository: Repository path for bash command execution

    Returns:
        Tuple of (has_pip_install, all_valid, validation_issues)
    """
    return await _validate_pip_install_commands_bashlex(script_content, available_files, check_file_existence, execute_bash_command, repository)


async def _validate_pip_install_commands_bashlex(script_content: str, available_files: List[str], check_file_existence: bool = False, execute_bash_command: Optional[Callable] = None, repository: Optional[str] = None, limit_depth: Optional[int] = None) -> Tuple[bool, bool, List[str]]:
    """Validate pip install commands using bashlex for robust parsing."""
    try:
        install_commands = find_install_commands_with_bashlex(script_content, limit_depth=limit_depth)
    except (ImportError, Exception):
        # If bashlex fails (e.g., syntax errors), use simple regex to check if pip commands exist
        import re

        has_pip = bool(re.search(r"\bpip\s+install\b", script_content) or re.search(r"\bpip3\s+install\b", script_content) or re.search(r"\bpython\s+-m\s+pip\s+install\b", script_content))
        if has_pip:
            return True, False, ["Script has syntax errors that prevent detailed validation"]
        else:
            return False, True, []

    pip_commands = [cmd for cmd in install_commands if cmd["command_type"] == "pip"]

    if not pip_commands:
        return False, True, []

    has_pip_install = True
    all_valid = True
    validation_issues = []

    # Map of valid pip install flags to whether they require a value
    pip_flag_specs = {
        # flags with arguments
        "-r": True,
        "--requirement": True,
        "-c": True,
        "--constraint": True,
        "-e": True,
        "--editable": True,
        "-t": True,
        "--target": True,
        "--platform": True,
        "--python-version": True,
        "--implementation": True,
        "--abi": True,
        "--root": True,
        "--prefix": True,
        "--src": True,
        "--upgrade-strategy": True,
        "-C": True,
        "--config-settings": True,
        "--global-option": True,
        "--no-binary": True,
        "--only-binary": True,
        "--progress-bar": True,
        "--root-user-action": True,
        "--report": True,
        "--group": True,
        "-i": True,
        "--index-url": True,
        "--extra-index-url": True,
        "-f": True,
        "--find-links": True,
        # boolean flags
        "--no-deps": False,
        "--pre": False,
        "--dry-run": False,
        "-U": False,
        "--upgrade": False,
        "--force-reinstall": False,
        "-I": False,
        "--ignore-installed": False,
        "--ignore-requires-python": False,
        "--no-build-isolation": False,
        "--use-pep517": False,
        "--check-build-dependencies": False,
        "--break-system-packages": False,
        "--compile": False,
        "--no-compile": False,
        "--no-warn-script-location": False,
        "--no-warn-conflicts": False,
        "--prefer-binary": False,
        "--require-hashes": False,
        "--no-clean": False,
        "--no-index": False,
        "--user": False,
    }

    async def validate_requirements_file(file_path: str) -> Tuple[bool, List[str]]:
        """Validate a requirements file by checking its content."""
        if not execute_bash_command or not repository:
            # Fallback to basic file existence check
            normalized_file = file_path[2:] if file_path.startswith("./") else file_path
            normalized_files = {f[2:] if f.startswith("./") else f for f in available_files}
            if normalized_file not in normalized_files:
                return False, [f"Requirements file {file_path} not found in configuration files"]
            return True, []

        try:
            result = await execute_bash_command(script=f"cat {shlex.quote(file_path)}", repository=repository)
            # Extract stdout content
            if "stdout:\n" in result:
                content = result.split("stdout:\n")[1]
                if "\nstderr:" in content:
                    content = content.split("\nstderr:")[0]
                content = content.strip()
            else:
                content = result.strip()

            if result.startswith("ERROR") or not content:
                return False, [f"Requirements file {file_path} could not be read or does not exist"]

            is_valid, content_issues = validate_requirements_file_content(content)
            if not is_valid:
                issues = [f"Requirements file {file_path} has invalid content:"] + [f"  {issue}" for issue in content_issues]
                return False, issues

            return True, []
        except Exception as e:
            return False, [f"Error validating requirements file {file_path}: {str(e)}"]

    for cmd_info in pip_commands:
        args = cmd_info["arguments"]

        # Skip validation if command contains variables or is in conditional context
        if should_skip_file_validation_bashlex(cmd_info):
            continue

        # Must have at least one requirement (non-flag) after install
        args_after = [tok for tok in args if not tok.startswith("-")]
        if not args_after:
            if not any(tok.startswith("--") and "=" in tok for tok in args):
                validation_issues.append(f"pip install command missing package requirements: {cmd_info['command_text']}")
                all_valid = False
                continue

        # Validate each flag token
        i = 0
        while i < len(args):
            tok = args[i]
            if not tok.startswith("-"):
                i += 1
                continue

            # Handle --flag=value form
            if tok.startswith("--") and "=" in tok:
                flag, val = tok.split("=", 1)
                if flag not in pip_flag_specs or not pip_flag_specs[flag]:
                    validation_issues.append(f"Unknown or invalid pip install flag: {flag}")
                    all_valid = False
                    break
                if val == "":
                    validation_issues.append(f"pip install flag {flag} has empty value")
                    all_valid = False
                    break

                # Validate requirements files (only for literal string paths, not variables/expressions)
                if flag in ["-r", "--requirement"] and check_file_existence and not _contains_variable_patterns(val):
                    try:
                        is_valid_file, file_issues = await validate_requirements_file(val)
                        if not is_valid_file:
                            validation_issues.extend(file_issues)
                            all_valid = False
                    except Exception as e:
                        validation_issues.append(f"Error validating requirements file {val}: {str(e)}")
                        all_valid = False

                i += 1
                continue

            # Separate flag token
            if tok not in pip_flag_specs:
                validation_issues.append(f"Unknown pip install flag: {tok}")
                all_valid = False
                break

            requires_value = pip_flag_specs[tok]
            if requires_value:
                if i + 1 >= len(args) or args[i + 1].startswith("-"):
                    validation_issues.append(f"pip install flag {tok} requires a value")
                    all_valid = False
                    break

                value = args[i + 1]
                # Validate requirements files (only for literal string paths, not variables/expressions)
                if tok in ["-r", "--requirement"] and check_file_existence and not _contains_variable_patterns(value):
                    try:
                        is_valid_file, file_issues = await validate_requirements_file(value)
                        if not is_valid_file:
                            validation_issues.extend(file_issues)
                            all_valid = False
                    except Exception as e:
                        validation_issues.append(f"Error validating requirements file {value}: {str(e)}")
                        all_valid = False

                i += 2  # skip flag + its value
            else:
                i += 1

            if not all_valid:
                break

    return has_pip_install, all_valid, validation_issues


def validate_poetry_install_commands(script_content: str) -> Tuple[bool, bool, List[str]]:
    """Validate poetry install commands in script content.

    Args:
        script_content: The bash script content to analyze

    Returns:
        Tuple of (has_poetry_install, all_valid, validation_issues)
    """
    return _validate_poetry_install_commands_bashlex(script_content)


def _validate_poetry_install_commands_bashlex(script_content: str) -> Tuple[bool, bool, List[str]]:
    """Validate poetry install commands using bashlex for robust parsing."""
    try:
        install_commands = find_install_commands_with_bashlex(script_content)
    except (ImportError, Exception):
        # If bashlex fails (e.g., syntax errors), use simple regex to check if poetry commands exist
        import re

        executable_content = "\n".join(extract_executable_lines(script_content))
        has_poetry = bool(re.search(r"\bpoetry\s+install\b", executable_content, flags=re.MULTILINE))
        if has_poetry:
            return True, False, ["Script has syntax errors that prevent detailed validation"]
        else:
            return False, True, []

    poetry_commands = [cmd for cmd in install_commands if cmd["command_type"] == "poetry"]

    if not poetry_commands:
        return False, True, []

    has_poetry_install = True
    all_valid = True
    validation_issues = []

    # All flags that Poetry install accepts
    poetry_flag_specs = {
        # value-taking flags
        "--without": True,
        "--with": True,
        "--only": True,
        "--extras": True,
        "-E": True,
        # boolean flags
        "--only-root": False,
        "--sync": False,
        "--no-root": False,
        "--no-directory": False,
        "--dry-run": False,
        "--all-extras": False,
        "--all-groups": False,
        "--compile": False,
        "--no-interaction": False,
        "--quiet": False,
        "-q": False,
        "--verbose": False,
        "-v": False,
        "--ansi": False,
        "--no-ansi": False,
        "--remove-untracked": False,
    }

    for cmd_info in poetry_commands:
        args = cmd_info["arguments"]

        # Skip validation if command contains variables or is in conditional context
        if should_skip_file_validation_bashlex(cmd_info):
            continue

        # Validate each flag token
        i = 0
        while i < len(args):
            tok = args[i]
            if not tok.startswith("-"):
                i += 1
                continue

            # Handle --flag=value form
            if tok.startswith("--") and "=" in tok:
                flag, val = tok.split("=", 1)
                if flag not in poetry_flag_specs or not poetry_flag_specs[flag]:
                    validation_issues.append(f"Unknown or invalid poetry install flag: {flag}")
                    all_valid = False
                    break
                if val == "":
                    validation_issues.append(f"poetry install flag {flag} has empty value")
                    all_valid = False
                    break
                i += 1
                continue

            # Separate flag token
            if tok not in poetry_flag_specs:
                validation_issues.append(f"Unknown poetry install flag: {tok}")
                all_valid = False
                break

            requires_value = poetry_flag_specs[tok]
            if requires_value:
                if i + 1 >= len(args) or args[i + 1].startswith("-"):
                    validation_issues.append(f"poetry install flag {tok} requires a value")
                    all_valid = False
                    break
                i += 2  # skip flag + its value
            else:
                i += 1

            if not all_valid:
                break

    return has_poetry_install, all_valid, validation_issues
