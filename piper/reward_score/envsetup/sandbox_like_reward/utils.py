import aiohttp


def truncate(text: str, max_length) -> str:
    """Truncate text to a maximum length."""
    if max_length < 0 or len(text) <= max_length:
        return text
    return text[:max_length] + f"\n[truncated to {max_length} characters]"


async def default_execute_bash_command(bash_command: str, repository: str) -> dict[str, str]:
    """Default implementation of execute_bash_command using the provided endpoint."""
    async with aiohttp.ClientSession() as session:
        payload = {"bash_command": bash_command, "repository": repository}
        async with session.post(
            "https://envbench-explorer.wlko.me/execute", json=payload, headers={"Content-Type": "application/json"}
        ) as response:
            result = await response.json()
            return {
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "exit_code": result.get("exit_code", 1),
                "success": result.get("success", False),
            }
            
PYTHON_CONTEXT_COMMANDS = [
    # Repository structure
    "tree -a -L 3 --filelimit 100 || ls -R",  # Fallback to ls -R if tree not available
    # Core setup files - direct read
    # "for f in setup.py pyproject.toml setup.cfg tox.ini; do "
    # 'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    # Requirements files - direct read
    # 'for f in requirements.txt requirements/*.txt; do if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    # Documentation - direct read
    "for f in README.md INSTALL.md SETUP.md docs/INSTALL.md docs/SETUP.md; do "
    'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    # Find and show all Python build files
    "find . -type f \\( "
    '-name "*requirements*.txt" -o -name "setup.py" -o -name "pyproject.toml" -o -name "setup.cfg" -o -name "tox.ini" '
    '\\) | while read f; do echo -e "\\n=== $f ==="; cat "$f"; done',
    # Python version info
    'find . -type f -name "*.py" -exec grep -l "python_version\\|python_requires" {} \\;',
    # Environment files
    'find . -type f \\( -name ".env*" -o -name "*.env" -o -name "Dockerfile*" \\) | '
    'while read f; do echo -e "\\n=== $f ==="; cat "$f"; done',
    # Docker files - direct read
    # "for f in Dockerfile docker-compose.yml docker-compose.yaml; do "
    # 'if [ -f "$f" ]; then echo -e "\\n=== $f ==="; cat "$f"; fi; done',
    # Python setup instructions in docs
    # 'find . -type f -name "*.md" -exec grep -i "python\|pip\|requirements\|virtualenv\|venv" {} \\;',
    # Additional Python files that might contain dependencies
    # 'find . -maxdepth 3 -type f -name "__init__.py" | while read f; do echo -e "\\n=== $f ==="; cat "$f"; done',
]


async def gather_repo_exploration(repo_name: str) -> str:
    """Gather repository exploration results."""
    # Combine all commands into a single bash script to avoid multiple requests
    combined_script = "#!/bin/bash\n\n"
    combined_script += "echo '=== REPOSITORY EXPLORATION START ==='\n\n"
    
    for i, command in enumerate(PYTHON_CONTEXT_COMMANDS, 1):
        combined_script += f"echo '--- Command {i} ---'\n"
        combined_script += f"echo 'Command: {command}'\n"
        combined_script += "echo 'Exit code:'\n"
        combined_script += f"({command}) 2>&1\n"
        combined_script += "echo 'Exit code: $?'\n"
        combined_script += f"echo '--- End Command {i} ---'\n\n"
    
    combined_script += "echo '=== REPOSITORY EXPLORATION END ==='\n"
    
    try:
        result = await default_execute_bash_command(combined_script, repo_name)
        exploration_text = f"Repository exploration for {repo_name}:\n"
        exploration_text += f"{truncate(result['stdout'], 8000)}\n"
        exploration_text += f"{truncate(result['stderr'], 2000)}"
        return exploration_text
    except Exception as e:
        return f"Error during repository exploration for {repo_name}: {str(e)}"
