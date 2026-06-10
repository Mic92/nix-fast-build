import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import IO

from .results import Result, ResultType

logger = logging.getLogger(__name__)


def strip_ansi(raw: str) -> str:
    """
    Removes ANSI escape sequences as defined by ECMA-048 in
    https://www.ecma-international.org/wp-content/uploads/ECMA-48_5th_edition_june_1991.pdf
    """
    return re.compile(
        r"""
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )""",
        re.VERBOSE,
    ).sub("", raw)


def get_ci_summary_file() -> Path | None:
    """Get the CI summary file path from environment.

    Supports GitHub Actions, Gitea Actions, and Forgejo Actions.
    """
    # Check for step summary environment variables from various CI systems
    for env_var in [
        "GITHUB_STEP_SUMMARY",
        "GITEA_STEP_SUMMARY",
        "FORGEJO_STEP_SUMMARY",
    ]:
        summary_path = os.environ.get(env_var)
        if summary_path:
            return Path(summary_path)
    return None


def format_failed_results(failed_results: dict[ResultType, list[Result]]) -> list[str]:
    """Format failed results section of CI summary."""
    lines: list[str] = []

    if not failed_results:
        return lines

    # Failed evaluations
    if ResultType.EVAL in failed_results:
        lines.append("\n### Failed Evaluations\n")
        for result in failed_results[ResultType.EVAL]:
            lines.append(f"**`{result.attr}`**\n")
            if result.error:
                # Check if error is multi-line (backtrace), if so collapse it
                error_lines = result.error.strip().split("\n")
                if len(error_lines) > 3:
                    # Long error message, collapse it
                    lines.append("<details>")
                    lines.append("<summary>Error details</summary>\n")
                    lines.append("```")
                    lines.extend(error_lines)
                    lines.append("```")
                    lines.append("</details>\n")
                else:
                    # Short error, display inline
                    lines.append(f"Error: {result.error}\n")

    # Failed builds with detailed logs
    if ResultType.BUILD in failed_results:
        lines.append("\n### Failed Builds\n")
        for result in failed_results[ResultType.BUILD]:
            lines.append(f"\n**{result.attr}** (duration: {result.duration:.2f}s)\n")
            if result.log_output:
                # Truncate very long logs (keep last 100 lines)
                log_lines = strip_ansi(result.log_output).strip().split("\n")
                if len(log_lines) > 100:
                    log_lines = [
                        "... (truncated, showing last 100 lines) ...",
                        *log_lines[-100:],
                    ]
                lines.append("\n<details>")
                lines.append(f"<summary>Build Log ({len(log_lines)} lines)</summary>\n")
                lines.append("```")
                lines.extend(log_lines)
                lines.append("```")
                lines.append("</details>\n")
            elif result.error:
                # If no log available, show error message
                lines.append(f"Error: {result.error}\n")

    # Other failed operations (uploads, downloads, etc.)
    for result_type in [
        ResultType.UPLOAD,
        ResultType.DOWNLOAD,
        ResultType.CACHIX,
        ResultType.ATTIC,
        ResultType.NIKS3,
    ]:
        if result_type in failed_results:
            type_name = result_type.name.title()
            lines.append(f"\n### Failed {type_name}s\n")
            for result in failed_results[result_type]:
                lines.append(f"**`{result.attr}`**\n")
                if result.error:
                    lines.append(f"Error: {result.error}\n")

    return lines


def format_successful_results(
    success_results: dict[ResultType, list[Result]],
) -> list[str]:
    """Format successful results section of CI summary."""
    lines: list[str] = []

    if not success_results:
        return lines

    lines.append("\n## Successful Operations\n")

    # Successful builds
    if ResultType.BUILD in success_results:
        build_count = len(success_results[ResultType.BUILD])
        lines.append("\n<details>")
        lines.append(f"<summary>Built {build_count} packages</summary>\n")
        lines.extend(
            [
                f"- {result.attr} ({result.duration:.2f}s)"
                for result in success_results[ResultType.BUILD]
            ]
        )
        lines.append("</details>\n")

    # Successful evaluations
    if ResultType.EVAL in success_results:
        eval_count = len(success_results[ResultType.EVAL])
        lines.append("\n<details>")
        lines.append(f"<summary>Evaluated {eval_count} attributes</summary>\n")
        lines.extend(
            [f"- {result.attr}" for result in success_results[ResultType.EVAL]]
        )
        lines.append("</details>\n")

    # Other successful operations
    for result_type in [
        ResultType.UPLOAD,
        ResultType.DOWNLOAD,
        ResultType.CACHIX,
        ResultType.ATTIC,
        ResultType.NIKS3,
    ]:
        if result_type in success_results:
            count = len(success_results[result_type])
            type_name = result_type.name.title()
            lines.append("\n<details>")
            lines.append(f"<summary>{type_name}: {count} successful</summary>\n")
            lines.extend(
                [f"- {result.attr}" for result in success_results[result_type]]
            )
            lines.append("</details>\n")

    return lines


def write_ci_summary(summary_file: Path, results: list[Result], rc: int) -> None:
    """Write CI job summary in markdown format.

    Supports GitHub Actions, Gitea Actions, and Forgejo Actions.
    """
    # Group results by success/failure and type
    failed_results: dict[ResultType, list[Result]] = defaultdict(list)
    success_results: dict[ResultType, list[Result]] = defaultdict(list)

    for r in results:
        if r.success:
            success_results[r.result_type].append(r)
        else:
            failed_results[r.result_type].append(r)

    # Build the markdown content
    lines = []
    lines.append("# nix-fast-build Results\n")

    # Overall status with summary counts
    total_success = sum(len(results) for results in success_results.values())
    total_failed = sum(len(results) for results in failed_results.values())

    if rc == 0:
        lines.append(f"## ✅ All Checks Passed ({total_success} successful)\n")
    else:
        lines.append(
            f"## ❌ Build Failed ({total_failed} failed, {total_success} successful)\n"
        )

    # Show failures first - they're what users need to act on
    lines.extend(format_failed_results(failed_results))

    # Show successful operations - collapsed by default for cleaner view
    lines.extend(format_successful_results(success_results))

    # Write to file
    try:
        with summary_file.open("a") as f:
            f.write("\n".join(lines))
        logger.info(f"CI summary written to {summary_file}")
    except OSError as e:
        logger.warning(f"Failed to write CI summary to {summary_file}: {e}")


def capitalize_first_letter(s: str) -> str:
    return s[0].upper() + s[1:].lower()


def dump_json(file: IO[str], results: list[Result]) -> None:
    json.dump(
        {
            "results": [
                {
                    "type": r.result_type.name,
                    "attr": r.attr,
                    "success": r.success,
                    "duration": r.duration,
                    "error": r.error,
                    **({"outputs": r.outputs} if r.outputs is not None else {}),
                }
                for r in results
            ]
        },
        file,
        indent=2,
        sort_keys=True,
    )


def dump_junit_xml(file: IO[str], suite_name: str, build_results: list[Result]) -> None:
    """
    Generates a JUnit XML report based on the results of Nix builds.

    Args:
        suite_name: Human-readable name for the test suite.
        build_results: A list of Result instances containing build result data.
        file: The output file where the XML report will be written.
    """
    testsuites = ET.Element("testsuites")
    testsuite = ET.SubElement(
        testsuites,
        "testsuite",
        {
            "name": suite_name,
            "tests": str(len(build_results)),
            "failures": str(sum(1 for r in build_results if not r.success)),
        },
    )

    for result in build_results:
        testcase = ET.SubElement(
            testsuite,
            "testcase",
            {
                "classname": capitalize_first_letter(result.result_type.name),
                "name": result.attr,
                "time": str(result.duration),
            },
        )

        if not result.success:
            failure = ET.SubElement(
                testcase,
                "failure",
                {
                    "message": result.error or "<no message>",
                    "type": "BuildFailure",
                },
            )
            # Strip ANSI escapes and XML-illegal control characters
            # (C0 range except TAB/LF/CR) from log output for valid XML.
            raw = strip_ansi(result.log_output or result.error or "")
            failure.text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)

    ET.ElementTree(testsuites).write(file, encoding="unicode")
