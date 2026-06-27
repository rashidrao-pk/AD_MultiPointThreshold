__all__ = [
    "dispatch_trainer",
    "main",
    "parse_args",
]


def __getattr__(name):
    """Load public training entry points only when requested."""
    if name in __all__:
        from . import training

        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
