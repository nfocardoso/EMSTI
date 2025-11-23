"""
H3 CLI Tool - Command-line interface for H3 analysis and reporting.

Usage:
    h3-report baseline.json h3_run.json
    h3-report baseline.json h3_run.json --json-summary results.json

Input JSON format:
{
    "name": "AdamW baseline",
    "final_accuracy": 89.94,
    "training_time_s": 615.2,
    "energy_joules": 36520.5,
    "epochs": 30
}
"""

import json
import sys
from typing import Dict, Any, Optional


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filepath}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {filepath}: {e}")
        sys.exit(1)


def compare_runs(baseline: Dict[str, Any], h3_run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare H3 run against baseline optimizer.

    Returns:
        Dict with comparison metrics and verdict.
    """
    # Extract metrics
    baseline_acc = baseline.get("final_accuracy", 0.0)
    h3_acc = h3_run.get("final_accuracy", 0.0)

    baseline_time = baseline.get("training_time_s", 1.0)
    h3_time = h3_run.get("training_time_s", 1.0)

    baseline_energy = baseline.get("energy_joules", 1.0)
    h3_energy = h3_run.get("energy_joules", 1.0)

    # Calculate differences
    accuracy_diff_pp = h3_acc - baseline_acc  # Percentage points
    time_reduction_pct = ((baseline_time - h3_time) / baseline_time) * 100
    energy_reduction_pct = ((baseline_energy - h3_energy) / baseline_energy) * 100

    # Generate verdict
    verdict = _generate_verdict(
        accuracy_diff_pp,
        time_reduction_pct,
        energy_reduction_pct
    )

    return {
        "baseline_name": baseline.get("name", "Baseline"),
        "h3_name": h3_run.get("name", "H3 Run"),
        "baseline_accuracy": baseline_acc,
        "h3_accuracy": h3_acc,
        "accuracy_diff_pp": accuracy_diff_pp,
        "baseline_time_s": baseline_time,
        "h3_time_s": h3_time,
        "time_reduction_pct": time_reduction_pct,
        "baseline_energy_j": baseline_energy,
        "h3_energy_j": h3_energy,
        "energy_reduction_pct": energy_reduction_pct,
        "verdict": verdict,
    }


def _generate_verdict(
    accuracy_diff_pp: float,
    time_reduction_pct: float,
    energy_reduction_pct: float
) -> str:
    """
    Generate human-readable verdict based on trade-offs.

    Criteria:
    - EXCELLENT: Accuracy improvement + efficiency gains
    - GOOD: Competitive accuracy + strong efficiency
    - ACCEPTABLE: Minor accuracy loss + major efficiency
    - POOR: Accuracy loss + weak efficiency
    """
    # Excellent: Win-win scenario
    if accuracy_diff_pp >= 0 and time_reduction_pct >= 10 and energy_reduction_pct >= 10:
        return "EXCELLENT - Accuracy improved AND efficiency gains achieved"

    # Good: Competitive with strong efficiency
    if accuracy_diff_pp >= -1.0 and time_reduction_pct >= 15 and energy_reduction_pct >= 15:
        return "GOOD - Competitive accuracy with strong efficiency gains"

    # Acceptable: Trade-off justified
    if accuracy_diff_pp >= -2.0 and time_reduction_pct >= 20 and energy_reduction_pct >= 20:
        return "ACCEPTABLE - Minor accuracy trade-off for major efficiency"

    # Poor: Not worth the trade-off
    if accuracy_diff_pp < -2.0 or time_reduction_pct < 5:
        return "POOR - Accuracy loss or insufficient efficiency gains"

    # Default: Neutral
    return "NEUTRAL - Mixed results, case-by-case evaluation needed"


def print_comparison(comparison: Dict[str, Any]) -> None:
    """Print formatted comparison report."""
    print("=" * 80)
    print("H3 OPTIMIZER COMPARISON REPORT")
    print("=" * 80)
    print()

    print(f"Baseline: {comparison['baseline_name']}")
    print(f"H3 Run:   {comparison['h3_name']}")
    print()

    print("-" * 80)
    print("ACCURACY")
    print("-" * 80)
    print(f"  Baseline:  {comparison['baseline_accuracy']:.2f}%")
    print(f"  H3:        {comparison['h3_accuracy']:.2f}%")

    acc_diff = comparison['accuracy_diff_pp']
    symbol = "✅" if acc_diff >= 0 else "⚠️"
    sign = "+" if acc_diff >= 0 else ""
    print(f"  Difference: {symbol} {sign}{acc_diff:.2f}pp")
    print()

    print("-" * 80)
    print("TRAINING TIME")
    print("-" * 80)
    print(f"  Baseline:  {comparison['baseline_time_s']:.1f}s")
    print(f"  H3:        {comparison['h3_time_s']:.1f}s")

    time_red = comparison['time_reduction_pct']
    symbol = "✅" if time_red >= 10 else "⚠️"
    sign = "" if time_red >= 0 else "+"
    print(f"  Reduction:  {symbol} {sign}{time_red:.1f}%")
    print()

    print("-" * 80)
    print("ENERGY CONSUMPTION")
    print("-" * 80)
    print(f"  Baseline:  {comparison['baseline_energy_j']:.1f}J")
    print(f"  H3:        {comparison['h3_energy_j']:.1f}J")

    energy_red = comparison['energy_reduction_pct']
    symbol = "✅" if energy_red >= 10 else "⚠️"
    sign = "" if energy_red >= 0 else "+"
    print(f"  Reduction:  {symbol} {sign}{energy_red:.1f}%")
    print()

    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"{comparison['verdict']}")
    print("=" * 80)


def h3_report_cli() -> None:
    """
    Main CLI entry point for h3-report command.

    Usage:
        h3-report baseline.json h3_run.json
        h3-report baseline.json h3_run.json --json-summary results.json
    """
    args = sys.argv[1:]

    if len(args) < 2:
        print("Usage: h3-report baseline.json h3_run.json [--json-summary output.json]")
        print()
        print("Compare H3 optimizer results against baseline.")
        print()
        print("Arguments:")
        print("  baseline.json    JSON file with baseline optimizer results")
        print("  h3_run.json      JSON file with H3 optimizer results")
        print()
        print("Options:")
        print("  --json-summary FILE   Save comparison results to JSON file")
        print()
        print("JSON format:")
        print('  {"name": "...", "final_accuracy": 89.5, "training_time_s": 600,')
        print('   "energy_joules": 35000, "epochs": 30}')
        sys.exit(1)

    baseline_file = args[0]
    h3_file = args[1]

    # Check for optional JSON output
    json_output = None
    if len(args) >= 4 and args[2] == "--json-summary":
        json_output = args[3]

    # Load data
    baseline = load_json(baseline_file)
    h3_run = load_json(h3_file)

    # Compare
    comparison = compare_runs(baseline, h3_run)

    # Print report
    print_comparison(comparison)

    # Save JSON if requested
    if json_output:
        try:
            with open(json_output, 'w') as f:
                json.dump(comparison, f, indent=2)
            print()
            print(f"✅ Comparison saved to: {json_output}")
        except IOError as e:
            print()
            print(f"⚠️  Warning: Could not save JSON output: {e}")


def main():
    """Alias for h3_report_cli() for backward compatibility."""
    h3_report_cli()


if __name__ == "__main__":
    h3_report_cli()
