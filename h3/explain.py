"""
H3 Training Run Explanation Utilities
Convert raw logs into actionable insights
"""

import pandas as pd
from typing import Optional, Dict
from pathlib import Path


def explain_thermo_log(
    h3_log_path: str,
    baseline_log_path: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Generate human-readable explanation of H3 training run.

    Args:
        h3_log_path: Path to H3 Thermo-Audit CSV log
        baseline_log_path: Optional path to baseline CSV log for comparison
        verbose: Include detailed phase analysis

    Returns:
        Formatted explanation string

    Example:
        >>> explanation = explain_thermo_log("mnist_h3.csv", "mnist_adam.csv")
        >>> print(explanation)
        📊 H3 TRAINING RUN ANALYSIS
        ...
    """
    df_h3 = pd.read_csv(h3_log_path)

    if df_h3.empty:
        return "❌ Error: H3 log is empty"

    # Extract final metrics
    final_row = df_h3.iloc[-1]
    final_acc = float(final_row["val_acc"])
    final_energy = float(final_row["cumulative_energy_j"])
    final_info = float(final_row["cumulative_info_bits"])
    final_eta = final_info / final_energy if final_energy > 0 else 0.0

    # Calculate averages
    avg_keep = float(df_h3["keep_frac"].mean())
    avg_mix = float(df_h3["uniform_mix"].mean())

    # Phase analysis
    phases = df_h3["phase"].value_counts().to_dict()

    # Build explanation
    lines = []
    lines.append("=" * 70)
    lines.append("📊 H3 TRAINING RUN ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    lines.append("### FINAL METRICS")
    lines.append(f"  • Final Accuracy:     {final_acc:.2f}%")
    lines.append(f"  • Total Energy:       {final_energy/1000:.2f} kJ ({final_energy:.0f} J)")
    lines.append(f"  • Info Gain:          {final_info:.2f} bits")
    lines.append(f"  • Efficiency η:       {final_eta:.6f} bits/J")
    lines.append("")

    lines.append("### TRAINING STRATEGY")
    lines.append(f"  • Avg keep_frac:      {avg_keep*100:.1f}% (data retention)")
    lines.append(f"  • Avg uniform_mix:    {avg_mix*100:.1f}% (exploration)")
    lines.append(f"  • Total epochs:       {len(df_h3)}")
    lines.append("")

    if verbose:
        lines.append("### PHASE BREAKDOWN")
        for phase, count in sorted(phases.items()):
            lines.append(f"  • {phase:15s}: {count:2d} epochs ({count/len(df_h3)*100:.1f}%)")
        lines.append("")

    # Heuristic assessment
    lines.append("### ASSESSMENT")
    if avg_keep < 0.4:
        lines.append("  ⚠️  WARNING: Starvation mode (keep_frac < 40%)")
        lines.append("      → Model may have lost important feature information")
        lines.append("      → Recommend: Increase keep_frac to 0.45-0.55")
    elif avg_keep < 0.50:
        lines.append("  ℹ️  Aggressive regime (40-50% data retention)")
        lines.append("      → Good for energy savings, watch accuracy trade-off")
    elif avg_keep < 0.65:
        lines.append("  ✅ Moderate regime (50-65% data retention)")
        lines.append("      → Balanced trade-off between speed and accuracy")
    else:
        lines.append("  ℹ️  Conservative regime (>65% data retention)")
        lines.append("      → Accuracy prioritized over energy savings")
    lines.append("")

    # Comparison with baseline
    if baseline_log_path:
        df_base = pd.read_csv(baseline_log_path)
        base_final = df_base.iloc[-1]
        base_acc = float(base_final["val_acc"])
        base_energy = float(base_final["cumulative_energy_j"])
        base_info = float(base_final["cumulative_info_bits"])

        acc_delta = final_acc - base_acc
        energy_saved_pct = (1 - final_energy / base_energy) * 100 if base_energy > 0 else 0
        info_delta = final_info - base_info

        lines.append("### COMPARISON vs BASELINE")
        lines.append(f"  • Accuracy:       {acc_delta:+.2f}pp "
                    f"({'✅ Better' if acc_delta >= 0 else '⚠️ Lower'})")
        lines.append(f"  • Energy saved:   {energy_saved_pct:+.1f}% "
                    f"({'✅ Efficient' if energy_saved_pct > 0 else '❌ Worse'})")
        lines.append(f"  • Info gain:      {info_delta:+.2f} bits")
        lines.append("")

        # Green Score
        green_score = base_energy / final_energy if final_energy > 0 else 1.0
        lines.append(f"  🌱 Green Score:   {green_score:.2f}x "
                    f"({'sustainable!' if green_score > 1.1 else 'similar to baseline'})")
        lines.append("")

        # Verdict
        lines.append("### VERDICT")
        if acc_delta >= -0.5 and energy_saved_pct >= 5:
            lines.append("  ✅ SUCCESS: Accuracy maintained with energy savings")
        elif acc_delta >= -1.0 and energy_saved_pct >= 3:
            lines.append("  ⚠️  ACCEPTABLE: Small accuracy trade-off for efficiency")
        elif acc_delta < -1.5:
            lines.append("  ❌ NEEDS TUNING: Accuracy loss too high")
            lines.append("     → Increase keep_frac by 0.05-0.10")
            lines.append("     → Extend consolidation phase")
        else:
            lines.append("  ℹ️  MIXED RESULTS: Review trade-offs for your use case")
        lines.append("")

    lines.append("=" * 70)
    lines.append(f"Log analyzed: {Path(h3_log_path).name}")
    if baseline_log_path:
        lines.append(f"Baseline:     {Path(baseline_log_path).name}")
    lines.append("=" * 70)

    return "\n".join(lines)


def green_score(energy_baseline_j: float, energy_h3_j: float) -> float:
    """
    Calculate "Green Score" - how much more energy-efficient H3 is.

    Args:
        energy_baseline_j: Baseline energy consumption (Joules)
        energy_h3_j: H3 energy consumption (Joules)

    Returns:
        Green score (1.0 = equal, >1.0 = H3 more efficient)

    Example:
        >>> score = green_score(40000, 35000)
        >>> print(f"H3 is {score:.2f}x more efficient")
        H3 is 1.14x more efficient
    """
    if energy_h3_j <= 0:
        return 1.0
    return float(energy_baseline_j / energy_h3_j)


def compare_runs(log_paths: Dict[str, str], metric: str = "efficiency") -> pd.DataFrame:
    """
    Compare multiple training runs.

    Args:
        log_paths: Dict mapping run names to CSV paths
        metric: Metric to compare ('efficiency', 'accuracy', 'energy', 'speed')

    Returns:
        DataFrame with comparison

    Example:
        >>> results = compare_runs({
        ...     "Adam": "adam.csv",
        ...     "H3_aggressive": "h3_aggressive.csv",
        ...     "H3_safe": "h3_safe.csv"
        ... })
        >>> print(results)
    """
    data = []

    for name, path in log_paths.items():
        df = pd.read_csv(path)
        final = df.iloc[-1]

        data.append({
            "Run": name,
            "Final Accuracy": float(final["val_acc"]),
            "Total Energy (kJ)": float(final["cumulative_energy_j"]) / 1000,
            "Efficiency (bits/J)": float(final["efficiency_bits_per_j"]),
            "Epochs": len(df),
            "Avg keep_frac": float(df["keep_frac"].mean()),
        })

    comparison = pd.DataFrame(data)

    # Sort by selected metric
    if metric == "efficiency":
        comparison = comparison.sort_values("Efficiency (bits/J)", ascending=False)
    elif metric == "accuracy":
        comparison = comparison.sort_values("Final Accuracy", ascending=False)
    elif metric == "energy":
        comparison = comparison.sort_values("Total Energy (kJ)", ascending=True)

    return comparison
