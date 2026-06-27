"""Training module entry point."""

from .training import (
    dispatch_trainer,
    main,
    parse_args,
)

__all__ = [
    "dispatch_trainer",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
