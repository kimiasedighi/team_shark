"""SwarmRules: Ferienakademie 2026 shared-project framework."""

from .api import Action, Cell, Observation, Result, RuleError
from .engine import run
from .scenarios import info

__all__ = [
    "Action",
    "Cell",
    "Observation",
    "Result",
    "RuleError",
    "run",
    "info",
]

__version__ = "1.0.0"
