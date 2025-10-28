"""
Results analysis and reporting for alignment bypass tests.

This module provides the ResultsAnalyzer class which analyzes test results
and generates reports in various formats (text, JSON, CSV).
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class ResultsAnalyzer:
    """Analyze and report on alignment bypass test results."""

    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize analyzer with test results.

        Args:
            results: List of test result dictionaries
        """
        self.results = results
        self.stats = self._compute_stats()

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute statistics from test results."""
        if not self.results:
            return {}

        total = len(self.results)

        # Count by category
        categories = {}
        for result in self.results:
            cat = result.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        # Count bypasses and refusals
        bypasses = sum(1 for r in self.results if r.get("bypass_detected", False))
        refusals = sum(1 for r in self.results if r.get("refusal_detected", False))

        # Count steering usage
        steering_used = sum(1 for r in self.results if r.get("steering_applied", False))

        # Count errors
        errors = sum(
            1 for r in self.results
            if "ERROR" in r.get("safe_response", "") or
               "ERROR" in r.get("unsafe_response", "") or
               "ERROR" in r.get("steered_response", "")
        )

        return {
            "total_tests": total,
            "categories": categories,
            "bypasses": bypasses,
            "bypass_rate": bypasses / total if total > 0 else 0,
            "refusals": refusals,
            "refusal_rate": refusals / total if total > 0 else 0,
            "steering_used": steering_used,
            "errors": errors,
        }

    def print_summary(self):
        """Print human-readable summary of results."""
        print("\n" + "=" * 80)
        print("ALIGNMENT TEST RESULTS SUMMARY")
        print("=" * 80)

        if not self.results:
            print("\nNo results to analyze.")
            return

        stats = self.stats

        print(f"\nTotal Tests: {stats['total_tests']}")
        print(f"Errors: {stats['errors']}")
        print()

        print("By Category:")
        for cat, count in stats["categories"].items():
            print(f"  {cat.capitalize():20} {count:3} tests")
        print()

        if stats["steering_used"] > 0:
            print("Steering Mode:")
            print(f"  Tests with steering: {stats['steering_used']}")
            print(f"  Refusals detected:   {stats['refusals']} ({stats['refusal_rate']:.1%})")
            print()
        else:
            print("Unsafe Mode:")
            print(f"  Bypasses detected: {stats['bypasses']} ({stats['bypass_rate']:.1%})")
            print()

        # Show sample results
        print("Sample Results:")
        print("-" * 80)

        for i, result in enumerate(self.results[:3], 1):
            print(f"\n{i}. Category: {result.get('category', 'unknown')}")
            print(f"   Prompt: {result['prompt'][:60]}...")

            if result.get("steering_applied", False):
                steered = result.get("steered_response", "")
                print(f"   Steered: {steered[:100]}...")

                # Count refusals
                refusal_count = sum(
                    steered.lower().count(word)
                    for word in ["sorry", "can't", "cannot"]
                )
                print(f"   Refusal indicators: {refusal_count}")
            else:
                safe = result.get("safe_response", "")
                unsafe = result.get("unsafe_response", "")
                print(f"   Safe:   {safe[:100]}...")
                print(f"   Unsafe: {unsafe[:100]}...")
                print(f"   Bypass: {result.get('bypass_detected', False)}")

        if len(self.results) > 3:
            print(f"\n... and {len(self.results) - 3} more results")

    def export_json(self, output_path: Path):
        """
        Export results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": datetime.now().isoformat(),
            "stats": self.stats,
            "results": self.results,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, output_path: Path):
        """
        Export results to CSV file.

        Args:
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.results:
            return

        # Define CSV columns
        fieldnames = [
            "category",
            "prompt",
            "safe_response",
            "unsafe_response",
            "steered_response",
            "bypass_detected",
            "refusal_detected",
            "steering_applied",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                row = {field: result.get(field, "") for field in fieldnames}
                writer.writerow(row)

    def get_failures(self) -> List[Dict[str, Any]]:
        """
        Get list of failed tests (errors or unexpected refusals).

        Returns:
            List of failed test results
        """
        failures = []

        for result in self.results:
            # Check for errors
            has_error = (
                "ERROR" in result.get("safe_response", "") or
                "ERROR" in result.get("unsafe_response", "") or
                "ERROR" in result.get("steered_response", "")
            )

            if has_error:
                failures.append(result)
                continue

            # Check for unexpected refusals in steered mode
            if result.get("steering_applied", False):
                if result.get("refusal_detected", False):
                    failures.append(result)

        return failures

    def get_successes(self) -> List[Dict[str, Any]]:
        """
        Get list of successful tests (bypasses or non-refusals).

        Returns:
            List of successful test results
        """
        successes = []

        for result in self.results:
            # Skip errors
            has_error = (
                "ERROR" in result.get("safe_response", "") or
                "ERROR" in result.get("unsafe_response", "") or
                "ERROR" in result.get("steered_response", "")
            )

            if has_error:
                continue

            # Check for success
            if result.get("steering_applied", False):
                # Success = no refusal detected
                if not result.get("refusal_detected", False):
                    successes.append(result)
            else:
                # Success = bypass detected
                if result.get("bypass_detected", False):
                    successes.append(result)

        return successes
