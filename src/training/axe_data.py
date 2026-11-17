"""Compatibility facade for historical ``src.training.axe_data`` imports.

New code should import :mod:`src.training.embedding_data`.
"""

from src.training.embedding_data import *  # noqa: F401,F403
