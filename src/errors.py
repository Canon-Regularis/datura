"""The root of the project's exception tree.

Each module keeps its own specific error class; every one of them inherits from
``DaturaError``. A caller that wants to distinguish a bad config from a bad
manifest catches the specific class; a caller that only wants to separate this
project's failures from a genuine bug catches the base.
"""

from __future__ import annotations


class DaturaError(Exception):
    """Base for every failure this project raises deliberately."""
