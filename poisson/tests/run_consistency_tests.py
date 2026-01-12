#!/usr/bin/env python3
"""
Phase 1 Consistency Test Runner for pypoisson2.

Runs the consistency test suite and generates reports in various formats.
Supports selective test execution and report generation.

Usage:
    python run_consistency_tests.py --full              # Run all tests
    python run_consistency_tests.py --category A         # Run category A tests
    python run_consistency_tests.py --test G1            # Run specific test
    python run_consistency_tests.py --report report.html # Generate HTML report
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from io import StringIO
import unittest

# Add repo and test directories to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from test_phase1_consistency import (
    TestCategoryA_Basic,
    TestCategoryB_Solver,
    TestCategoryC_Depth,
    TestCategoryD_Extraction,
    TestCategoryE_Grid,
    TestCategoryF_EdgeCases,
    TestCategoryG_RealData,
    ConsistencyTestResult,
)


class TestReport:
    """Container for test report data."""

    def __init__(self):
        self.start_time = datetime.now()
        self.end_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.errors = []
        self.failures = []
        self.test_results = []
        self.duration = 0.0

    def finish(self):
        """Mark report as finished."""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()

    @property
    def pass_rate(self):
        """Calculate pass rate."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100


class JSONTestRunner(unittest.TextTestRunner):
    """Custom test runner that collects detailed results."""

    def __init__(self, report=None, **kwargs):
        super().__init__(**kwargs)
        self.report = report or TestReport()

    def run(self, test):
        """Run tests and collect results."""
        self.report.total_tests = test.countTestCases()

        # Use StringIO to capture output
        stream = StringIO()
        result = unittest.TextTestRunner(
            stream=stream,
            verbosity=self.verbosity,
            buffer=self.buffer
        ).run(test)

        # Collect results
        self.report.passed_tests = result.testsRun - len(result.failures) - len(result.errors)
        self.report.failed_tests = len(result.failures)
        self.report.skipped_tests = len(result.skipped)
        self.report.errors = [(str(test), traceback) for test, traceback in result.errors]
        self.report.failures = [(str(test), traceback) for test, traceback in result.failures]

        self.report.finish()

        # Print summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print(f"Total tests:   {self.report.total_tests}")
        print(f"Passed:        {self.report.passed_tests}")
        print(f"Failed:        {self.report.failed_tests}")
        print(f"Skipped:       {self.report.skipped_tests}")
        print(f"Pass rate:     {self.report.pass_rate:.1f}%")
        print(f"Duration:      {self.report.duration:.2f}s")
        print("=" * 70)

        return result


def get_test_suite(categories=None, test_id=None, test_data=None, depth=None):
    """
    Build a test suite based on command line arguments.

    Parameters
    ----------
    categories : list of str
        Test categories to run (e.g., ['A', 'B', 'G']).
    test_id : str
        Specific test ID to run (e.g., 'A1', 'G1').
    test_data : str
        Path to custom test data file (XYZ format).
    depth : int
        Reconstruction depth for custom test.

    Returns
    -------
    suite : unittest.TestSuite
        Configured test suite.
    """
    suite = unittest.TestSuite()

    # Map category letters to test classes
    category_map = {
        'A': TestCategoryA_Basic,
        'B': TestCategoryB_Solver,
        'C': TestCategoryC_Depth,
        'D': TestCategoryD_Extraction,
        'E': TestCategoryE_Grid,
        'F': TestCategoryF_EdgeCases,
        'G': TestCategoryG_RealData,
    }

    if test_id:
        # Run specific test
        category_letter = test_id[0].upper()
        test_class = category_map.get(category_letter)

        if test_class:
            test_method = f"test_{test_id}"
            if hasattr(test_class, test_method):
                suite.addTest(test_class(test_method))
            else:
                print(f"Warning: Test {test_id} not found")
        else:
            print(f"Warning: Category {category_letter} not found")

    elif categories:
        # Run specific categories
        for cat in categories:
            cat = cat.upper()
            test_class = category_map.get(cat)
            if test_class:
                suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))
            else:
                print(f"Warning: Category {cat} not found")

    else:
        # Run all tests
        for test_class in category_map.values():
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))

    return suite


def generate_markdown_report(report, output_path):
    """Generate a Markdown report."""
    lines = [
        "# Phase 1 Consistency Test Report",
        "",
        f"**Generated:** {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Duration:** {report.duration:.2f} seconds",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total Tests | {report.total_tests} |",
        f"| Passed | {report.passed_tests} |",
        f"| Failed | {report.failed_tests} |",
        f"| Skipped | {report.skipped_tests} |",
        f"| Pass Rate | **{report.pass_rate:.1f}%** |",
        "",
    ]

    if report.failures:
        lines.extend([
            "## Failed Tests",
            "",
        ])
        for test_name, traceback in report.failures:
            lines.extend([
                f"### {test_name}",
                "```",
                traceback,
                "```",
                "",
            ])

    if report.errors:
        lines.extend([
            "## Errors",
            "",
        ])
        for test_name, traceback in report.errors:
            lines.extend([
                f"### {test_name}",
                "```",
                traceback,
                "```",
                "",
            ])

    lines.extend([
        "## Test Categories",
        "",
        "| Category | Description |",
        "|----------|-------------|",
        "| A | Basic Consistency Tests |",
        "| B | Core Solver Parameters |",
        "| C | Depth Control Parameters |",
        "| D | Mesh Extraction Parameters |",
        "| E | Grid Output Tests |",
        "| F | Edge Cases |",
        "| G | Real Data (Horse Model) |",
        "",
        "## Success Criteria",
        "",
        "- **Strict Consistency (P0):** iso_value diff < 1e-6, vertex/face counts match",
        "- **Loose Consistency (P1):** iso_value diff < 1e-4, vertex count diff < 5",
        "- **Behavioral:** Manifold topology, symmetry properties",
    ])

    content = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"\nMarkdown report saved to: {output_path}")


def generate_json_report(report, output_path):
    """Generate a JSON report."""
    data = {
        'summary': {
            'total_tests': report.total_tests,
            'passed': report.passed_tests,
            'failed': report.failed_tests,
            'skipped': report.skipped_tests,
            'pass_rate': report.pass_rate,
            'duration_seconds': report.duration,
            'start_time': report.start_time.isoformat(),
            'end_time': report.end_time.isoformat() if report.end_time else None,
        },
        'failures': [
            {'test': name, 'traceback': tb}
            for name, tb in report.failures
        ],
        'errors': [
            {'test': name, 'traceback': tb}
            for name, tb in report.errors
        ],
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"JSON report saved to: {output_path}")


def generate_html_report(report, output_path):
    """Generate an HTML report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Phase 1 Consistency Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ display: flex; justify-content: space-between; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 15px; background: #f9f9f9; border-radius: 5px; flex: 1; margin: 0 5px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ color: #777; font-size: 0.9em; }}
        .pass {{ color: #4CAF50; }}
        .fail {{ color: #f44336; }}
        .skip {{ color: #FF9800; }}
        .test-item {{ padding: 10px; margin: 5px 0; border-left: 3px solid #ddd; }}
        .test-failed {{ border-left-color: #f44336; background: #ffebee; }}
        .test-error {{ border-left-color: #FF9800; background: #fff3e0; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        .timestamp {{ color: #777; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Phase 1 Consistency Test Report</h1>
        <p class="timestamp">Generated: {report.start_time.strftime('%Y-%m-%d %H:%M:%S')} | Duration: {report.duration:.2f}s</p>

        <div class="summary">
            <div class="metric">
                <div class="metric-value">{report.total_tests}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric">
                <div class="metric-value pass">{report.passed_tests}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric">
                <div class="metric-value fail">{report.failed_tests}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value skip">{report.skipped_tests}</div>
                <div class="metric-label">Skipped</div>
            </div>
            <div class="metric">
                <div class="metric-value {'pass' if report.pass_rate >= 95 else 'fail' if report.pass_rate < 80 else 'skip'}">{report.pass_rate:.1f}%</div>
                <div class="metric-label">Pass Rate</div>
            </div>
        </div>

        <h2>Failed Tests</h2>
"""

    if report.failures:
        for test_name, traceback in report.failures:
            html += f"""
        <div class="test-item test-failed">
            <strong>{test_name}</strong>
            <pre>{traceback}</pre>
        </div>
"""
    else:
        html += "<p>No failures!</p>"

    html += """
        <h2>Errors</h2>
"""

    if report.errors:
        for test_name, traceback in report.errors:
            html += f"""
        <div class="test-item test-error">
            <strong>{test_name}</strong>
            <pre>{traceback}</pre>
        </div>
"""
    else:
        html += "<p>No errors!</p>"

    html += """
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Phase 1 consistency tests for pypoisson2'
    )
    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Run all tests'
    )
    parser.add_argument(
        '--category', '-c',
        action='append',
        help='Test category to run (e.g., A, B, G). Can be specified multiple times.'
    )
    parser.add_argument(
        '--test', '-t',
        help='Specific test ID to run (e.g., A1, G1)'
    )
    parser.add_argument(
        '--report', '-r',
        help='Generate report at specified path (format based on extension: .md, .json, .html)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='results',
        help='Output directory for reports (default: results)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Build test suite
    suite = get_test_suite(
        categories=args.category,
        test_id=args.test
    )

    if suite.countTestCases() == 0:
        print("No tests selected. Use --category, --test, or --full.")
        return 1

    print(f"Running {suite.countTestCases()} test(s)...\n")

    # Run tests
    report = TestReport()
    runner = JSONTestRunner(report, verbosity=2 if args.verbose else 1)
    result = runner.run(suite)

    # Generate reports
    if args.report:
        report_path = Path(args.report)

        if report_path.suffix == '.md':
            generate_markdown_report(report, report_path)
        elif report_path.suffix == '.json':
            generate_json_report(report, report_path)
        elif report_path.suffix == '.html':
            generate_html_report(report, report_path)
        else:
            print(f"Unknown report format: {report_path.suffix}")
    else:
        # Generate default reports
        generate_markdown_report(report, output_dir / 'consistency_report.md')
        generate_json_report(report, output_dir / 'consistency_report.json')

    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
