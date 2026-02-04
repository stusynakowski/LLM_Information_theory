"""
LLM Information Theory Analysis Toolkit
A toolkit for analyzing information-theoretic properties of language model generation.
"""

__version__ = "0.1.0"

from .probability_extractor import ProbabilityExtractor
from .information_theory import compute_entropy, compute_shannon_information
from .intervention import InterventionManager

__all__ = [
    'ProbabilityExtractor',
    'compute_entropy',
    'compute_shannon_information',
    'InterventionManager'
]
