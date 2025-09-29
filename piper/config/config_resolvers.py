"""
Custom OmegaConf resolvers for arithmetic operations and other utilities.
"""

from omegaconf import DictConfig, OmegaConf


def register_arithmetic_resolvers():
    """Register custom arithmetic resolvers for OmegaConf."""

    # Basic arithmetic operations
    OmegaConf.register_new_resolver("sub", lambda a, b: a - b, replace=True)

    # Safe eval for simple expressions (be careful with this)
    def safe_eval(expr):
        """Safely evaluate simple arithmetic expressions."""
        # Only allow basic arithmetic operations and numbers
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in str(expr)):
            raise ValueError(f"Invalid characters in expression: {expr}")
        try:
            return eval(expr)
        except Exception as e:
            raise ValueError(f"Cannot evaluate expression '{expr}': {e}") from e

    OmegaConf.register_new_resolver("eval", safe_eval, replace=True)


def extract_tags(tags_config: DictConfig) -> list[str]:
    """Extract tags from a DictConfig. Order is preserved."""
    if not tags_config:
        return []
    res = []
    for _, value in tags_config.items():
        if isinstance(value, str):
            res.append(value)
        elif isinstance(value, list):
            res.extend(value)
        elif isinstance(value, DictConfig):
            res.extend(extract_tags(value))
    return res


def tags_to_name(tags_config: DictConfig) -> str:
    """Convert tags to a string. Order is preserved."""
    tags = extract_tags(tags_config)
    return "_".join(tags)


def register_all_resolvers():
    """Register all custom resolvers."""
    register_arithmetic_resolvers()
    OmegaConf.register_new_resolver("tags_to_name", tags_to_name, replace=True)
    OmegaConf.register_new_resolver("tags_to_list", extract_tags, replace=True)
    print("Custom OmegaConf resolvers registered successfully!")


def init_resolvers():
    """Initialize resolvers automatically when imported."""
    try:
        # Check if resolvers are already registered
        if not OmegaConf.has_resolver("add"):
            register_all_resolvers()
    except Exception as e:
        print(f"Warning: Could not register custom resolvers: {e}")


# Auto-register resolvers when module is imported
init_resolvers()


if __name__ == "__main__":
    register_all_resolvers()
