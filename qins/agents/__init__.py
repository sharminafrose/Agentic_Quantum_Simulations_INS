"""Agent registry. Importing this package registers all agents."""

from .base import AGENT_REGISTRY, Agent, TaskContext, register_agent
from . import (  # noqa: F401  (import for registration side effect)
    a00_model_spec, a01_mps_dmrg, a02_gs_ansatz, a03_aqc, a04_hw_calibration, a05_noise_model,
    a06_metrics, a07_assembly, a08_hw_execution, a09_dsf, a10_benchmark,
)

__all__ = ["AGENT_REGISTRY", "Agent", "TaskContext", "register_agent"]
