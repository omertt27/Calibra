"""
calibra.models — learnable components for world-model observability.

Public API:
    RobotJEPA        : Joint Embedding Predictive Architecture for robot demos
    RobotJEPAConfig  : hyperparameter dataclass
"""

from calibra.models.robot_jepa import RobotJEPA, RobotJEPAConfig

__all__ = ["RobotJEPA", "RobotJEPAConfig"]
