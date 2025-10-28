"""
Alignment testing framework for GPT-OSS models.

This package provides tools for testing whether safety alignment is present
in model weights or only in prompt formatting, and for testing steering vectors.

Modules:
    - tester: Comparison framework with steering support (AlignmentTester)
    - analysis: Results analysis and export (ResultsAnalyzer)
    - prompts: Test prompt collections
    - generate_steered: Custom generation with steering vectors
    - apply_steering: Layer patching utilities
    - compute_vectors: Steering vector computation
    - capture_activations: Activation capture utilities
"""

from scripts.alignment.tester import AlignmentTester
from scripts.alignment.analysis import ResultsAnalyzer
from scripts.alignment.prompts import (
    get_prompt_categories,
    get_prompt_stats,
)

__all__ = [
    "AlignmentTester",
    "ResultsAnalyzer",
    "get_prompt_categories",
    "get_prompt_stats",
]

__version__ = "2.0.0"
