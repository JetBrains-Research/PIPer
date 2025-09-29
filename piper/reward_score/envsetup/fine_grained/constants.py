import fnmatch
import os
from typing import List

# Pattern-based mapping instead of exact filename mapping
FNAME_PATTERNS_TO_DM_MAP = {
    "*requirements*.txt": "pip",
    "requirements.txt": "pip",
    "requirements-*.txt": "pip",
    "*/requirements/*.txt": "pip",
    "setup.py": "pip",
    "setup.cfg": "pip",
    "poetry.lock": "poetry",
    "pyproject.toml": "unclear",
    ".python-version": "unclear",
}

# Keep the original mapping for backward compatibility
FNAME_TO_DM_MAP = {
    "requirements.txt": "pip",
    "setup.py": "pip",
    "setup.cfg": "pip",
    "poetry.lock": "poetry",
    "pyproject.toml": "unclear",
    ".python-version": "unclear",
}

BUILD_BACKEND_TO_DM_MAP = {
    "setuptools.build_meta": "pip",
    "setuptools.build_meta:__legacy__": "pip",
    "poetry.core.masonry.api": "poetry",
    "poetry.masonry.api": "poetry",
    "poetry_dynamic_versioning.backend": "poetry",
}


def match_file_patterns(file_path: str) -> List[str]:
    """Match a file path against patterns and return matching dependency managers.

    Args:
        file_path: Path to the file (can be relative or with directories)

    Returns:
        List of dependency managers that match the file patterns
    """
    matches = []

    # Normalize the file path (remove leading ./ prefix if present)
    if file_path.startswith("./"):
        normalized_path = file_path[2:]
    else:
        normalized_path = file_path

    filename = os.path.basename(normalized_path)

    for pattern, dm in FNAME_PATTERNS_TO_DM_MAP.items():
        # Check if pattern matches the full normalized path
        if fnmatch.fnmatch(normalized_path, pattern):
            if dm not in matches:
                matches.append(dm)
        # Check if pattern matches just the filename (for patterns without directory separators)
        elif "/" not in pattern and fnmatch.fnmatch(filename, pattern):
            if dm not in matches:
                matches.append(dm)
        # Also check if the original file_path (before normalization) matches
        elif fnmatch.fnmatch(file_path, pattern):
            if dm not in matches:
                matches.append(dm)

    return matches


def categorize_file(file_path: str) -> str:
    """Categorize a file into a dependency manager category.

    Args:
        file_path: Path to the file

    Returns:
        Dependency manager category ('pip', 'poetry', 'unclear') or None if no match
    """
    matches = match_file_patterns(file_path)

    if not matches:
        return None

    # If multiple matches, prioritize in order: poetry, pip, unclear
    if "poetry" in matches:
        return "poetry"
    elif "pip" in matches:
        return "pip"
    else:
        return "unclear"
