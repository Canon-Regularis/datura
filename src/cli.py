"""Shared pieces of the command line entry points.

Eight modules are runnable and they all take the same handful of options. Defining
those once means ``--config`` behaves identically everywhere, and adding a global
flag is a change in one file.
"""

from __future__ import annotations

import argparse

from src.config import Config, load_config
from src.logging_config import configure

DEFAULT_CONFIG = "configs/base.yaml"


def parser_for(description: str | None) -> argparse.ArgumentParser:
    """A parser carrying the options every entry point accepts."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="dataset, audio and split settings",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="log debug detail")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="log warnings only")
    return parser


def add_variant_name(parser: argparse.ArgumentParser, default: str = "cnn") -> None:
    """Which result directory a trained variant writes to."""
    parser.add_argument("--name", default=default, help="result directory for this variant")


def prepare(args: argparse.Namespace) -> Config:
    """Turn parsed arguments into a validated config, with logging switched on.

    Directories are created here so no later stage has to check whether its output
    folder exists.
    """
    configure(verbose=args.verbose, quiet=args.quiet)
    cfg = load_config(args.config)
    cfg.paths.ensure()
    return cfg
