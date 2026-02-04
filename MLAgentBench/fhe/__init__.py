"""
FHE (Fully Homomorphic Encryption) challenge support for MLAgentBench.

This module provides:
- Challenge parsing from challenge.md files
- Docker-based execution for FHE solutions
- FHE-specific actions for agents
"""

from .challenge_parser import (
    FHEChallengeSpec,
    ChallengeType,
    Scheme,
    Library,
    parse_challenge,
)

__all__ = [
    "FHEChallengeSpec",
    "ChallengeType",
    "Scheme",
    "Library",
    "parse_challenge",
]
