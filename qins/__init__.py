"""qins -- multi-agent benchmark of quantum simulations of inelastic neutron scattering."""

import os as _os

# Qiskit's parallel transpile forks processes; forking after JAX started threads can deadlock.
_os.environ.setdefault("QISKIT_PARALLEL", "FALSE")

from .config import load_config
from .pipeline import Pipeline, PipelineResult

__all__ = ["load_config", "Pipeline", "PipelineResult"]
__version__ = "0.1.0"
