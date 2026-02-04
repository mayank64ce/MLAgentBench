"""
FHE Interpreters - Docker-based execution for FHE challenges.
"""

from .base import BaseInterpreter, ExecutionResult, ValidationResult
from .black_box import BlackBoxInterpreter
from .white_box import WhiteBoxInterpreter


def create_interpreter(spec, workspace_dir, build_timeout=300, run_timeout=600):
    """
    Factory function to create the appropriate interpreter for a challenge.

    Args:
        spec: FHEChallengeSpec from challenge parser
        workspace_dir: Directory for workspace files
        build_timeout: Docker build timeout in seconds
        run_timeout: Docker run timeout in seconds

    Returns:
        Appropriate interpreter instance
    """
    from ..challenge_parser import ChallengeType

    if spec.challenge_type == ChallengeType.BLACK_BOX:
        return BlackBoxInterpreter(spec, workspace_dir, build_timeout, run_timeout)
    elif spec.challenge_type == ChallengeType.WHITE_BOX_OPENFHE:
        return WhiteBoxInterpreter(spec, workspace_dir, build_timeout, run_timeout)
    elif spec.challenge_type == ChallengeType.ML_INFERENCE:
        # ML inference uses white box interpreter with fherma-validator
        return WhiteBoxInterpreter(spec, workspace_dir, build_timeout, run_timeout)
    else:
        # Default to white box for unknown types
        return WhiteBoxInterpreter(spec, workspace_dir, build_timeout, run_timeout)


__all__ = [
    "BaseInterpreter",
    "ExecutionResult",
    "ValidationResult",
    "BlackBoxInterpreter",
    "WhiteBoxInterpreter",
    "create_interpreter",
]
