"""
H3 Command-Line Interface
Tools for analyzing and comparing training runs
"""

import argparse
import sys
from pathlib import Path
from .explain import explain_thermo_log, compare_runs


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="H3 Thermodynamic Training Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single H3 run
  h3-report --h3-log mnist_h3.csv

  # Compare H3 vs baseline
  h3-report --h3-log mnist_h3.csv --baseline-log mnist_adam.csv

  # Compare multiple runs
  h3-report --compare run1.csv run2.csv run3.csv
        """
    )

    parser.add_argument(
        "--h3-log",
        type=str,
        help="Path to H3 Thermo-Audit CSV log"
    )

    parser.add_argument(
        "--baseline-log",
        type=str,
        help="Path to baseline CSV log (for comparison)"
    )

    parser.add_argument(
        "--compare",
        nargs="+",
        type=str,
        help="Compare multiple runs (provide paths to CSV logs)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include detailed phase analysis"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Save report to file (default: print to stdout)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.h3_log and not args.compare:
        parser.error("Must provide either --h3-log or --compare")

    if args.h3_log and args.compare:
        parser.error("Cannot use --h3-log and --compare together")

    # Generate report
    if args.h3_log:
        # Check file exists
        if not Path(args.h3_log).exists():
            print(f"❌ Error: File not found: {args.h3_log}", file=sys.stderr)
            sys.exit(1)

        if args.baseline_log and not Path(args.baseline_log).exists():
            print(f"❌ Error: File not found: {args.baseline_log}", file=sys.stderr)
            sys.exit(1)

        # Generate explanation
        report = explain_thermo_log(
            args.h3_log,
            args.baseline_log,
            verbose=args.verbose
        )

        if args.output:
            Path(args.output).write_text(report)
            print(f"✅ Report saved to: {args.output}")
        else:
            print(report)

    elif args.compare:
        # Check all files exist
        for path in args.compare:
            if not Path(path).exists():
                print(f"❌ Error: File not found: {path}", file=sys.stderr)
                sys.exit(1)

        # Generate comparison
        log_paths = {Path(p).stem: p for p in args.compare}
        comparison = compare_runs(log_paths)

        print("\n" + "=" * 70)
        print("📊 MULTI-RUN COMPARISON")
        print("=" * 70)
        print()
        print(comparison.to_string(index=False))
        print()
        print("=" * 70)

        if args.output:
            comparison.to_csv(args.output, index=False)
            print(f"✅ Comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
