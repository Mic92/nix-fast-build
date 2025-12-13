"""Integration test for GitHub Actions summary feature."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nix_fast_build import (
    Options,
    Result,
    ResultType,
    get_ci_summary_file,
    write_ci_summary,
)


def test_github_actions_workflow() -> None:
    """Test complete GitHub Actions workflow with environment variables."""
    original_actions = os.environ.get("GITHUB_ACTIONS")
    original_summary = os.environ.get("GITHUB_STEP_SUMMARY")

    try:
        with TemporaryDirectory() as d:
            summary_path = Path(d) / "github_summary.md"

            # Set up GitHub Actions environment
            os.environ["GITHUB_ACTIONS"] = "true"
            os.environ["GITHUB_STEP_SUMMARY"] = str(summary_path)

            # Create options
            opts = Options(
                flake_url="github:example/repo", flake_fragment="checks.x86_64-linux"
            )

            # Verify it picks up the environment variable
            ci_summary_file = get_ci_summary_file()
            assert ci_summary_file == summary_path

            # Simulate build results
            results = [
                Result(
                    result_type=ResultType.EVAL,
                    attr="package-a",
                    success=True,
                    duration=2.1,
                    error=None,
                ),
                Result(
                    result_type=ResultType.BUILD,
                    attr="package-a",
                    success=True,
                    duration=15.3,
                    error=None,
                ),
                Result(
                    result_type=ResultType.EVAL,
                    attr="package-b",
                    success=True,
                    duration=1.8,
                    error=None,
                ),
                Result(
                    result_type=ResultType.BUILD,
                    attr="package-b",
                    success=False,
                    duration=8.2,
                    error="build exited with 1",
                    log_output=(
                        "error: builder for '/nix/store/xxx-package-b.drv' failed with exit code 1:\n"
                        "last 25 lines of build log:\n"
                        "> building\n"
                        "> checking for compiler\n"
                        "> error: missing dependency: libfoo\n"
                        "> build failed"
                    ),
                ),
                Result(
                    result_type=ResultType.UPLOAD,
                    attr="package-a",
                    success=True,
                    duration=3.5,
                    error=None,
                ),
            ]

            # Write the summary
            write_ci_summary(summary_path, opts, results, rc=1)

            # Verify the summary was written
            assert summary_path.exists()
            content = summary_path.read_text()

            # Verify content includes expected sections
            assert "# nix-fast-build Results" in content
            assert "❌ Build Failed" in content
            assert "1 failed, 4 successful" in content
            assert "### Failed Builds" in content
            assert "## Successful Operations" in content

            # Verify failed build details
            assert "**package-b**" in content
            assert "missing dependency: libfoo" in content
            assert "<details>" in content
            assert "Build Log" in content

            # Verify successful operations are listed
            assert "Built 1 packages" in content
            assert "Evaluated 2 attributes" in content
            assert "Upload: 1 successful" in content

    finally:
        # Restore original environment
        if original_actions is None:
            os.environ.pop("GITHUB_ACTIONS", None)
        else:
            os.environ["GITHUB_ACTIONS"] = original_actions
        if original_summary is None:
            os.environ.pop("GITHUB_STEP_SUMMARY", None)
        else:
            os.environ["GITHUB_STEP_SUMMARY"] = original_summary


def test_long_log_truncation() -> None:
    """Test that very long logs are truncated."""
    with TemporaryDirectory() as d:
        summary_file = Path(d) / "summary.md"
        opts = Options(flake_url=".#checks", flake_fragment="checks")

        # Create a log with more than 100 lines
        long_log = "\n".join([f"log line {i}" for i in range(150)])

        results = [
            Result(
                result_type=ResultType.BUILD,
                attr="test-package",
                success=False,
                duration=5.3,
                error="build failed",
                log_output=long_log,
            ),
        ]

        write_ci_summary(summary_file, opts, results, rc=1)

        content = summary_file.read_text()

        # Check for truncation message
        assert "truncated, showing last 100 lines" in content

        # Verify only last 100 lines + truncation message are present
        assert (
            "log line 50" in content
        )  # Should be present (line 50 onwards, first of last 100)
        assert "log line 49" not in content  # Should be truncated
        assert "log line 149" in content  # Last line should be present
