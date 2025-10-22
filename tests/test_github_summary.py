import os
from pathlib import Path
from tempfile import TemporaryDirectory

from nix_fast_build import (
    Options,
    Result,
    ResultType,
    get_github_summary_file,
    is_github_actions,
    write_github_summary,
)


def test_is_github_actions() -> None:
    """Test GitHub Actions detection."""
    # Save original value
    original = os.environ.get("GITHUB_ACTIONS")
    
    try:
        # Test with GITHUB_ACTIONS set to true
        os.environ["GITHUB_ACTIONS"] = "true"
        assert is_github_actions() is True
        
        # Test with GITHUB_ACTIONS set to false
        os.environ["GITHUB_ACTIONS"] = "false"
        assert is_github_actions() is False
        
        # Test without GITHUB_ACTIONS
        del os.environ["GITHUB_ACTIONS"]
        assert is_github_actions() is False
    finally:
        # Restore original value
        if original is None:
            os.environ.pop("GITHUB_ACTIONS", None)
        else:
            os.environ["GITHUB_ACTIONS"] = original


def test_get_github_summary_file_explicit() -> None:
    """Test explicit summary file path."""
    opts = Options(github_summary="/tmp/summary.md")
    result = get_github_summary_file(opts)
    assert result == Path("/tmp/summary.md")


def test_get_github_summary_file_from_env() -> None:
    """Test summary file from GITHUB_STEP_SUMMARY env var."""
    original_actions = os.environ.get("GITHUB_ACTIONS")
    original_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    
    try:
        os.environ["GITHUB_ACTIONS"] = "true"
        os.environ["GITHUB_STEP_SUMMARY"] = "/tmp/github_summary.md"
        
        opts = Options()
        result = get_github_summary_file(opts)
        assert result == Path("/tmp/github_summary.md")
    finally:
        # Restore original values
        if original_actions is None:
            os.environ.pop("GITHUB_ACTIONS", None)
        else:
            os.environ["GITHUB_ACTIONS"] = original_actions
        if original_summary is None:
            os.environ.pop("GITHUB_STEP_SUMMARY", None)
        else:
            os.environ["GITHUB_STEP_SUMMARY"] = original_summary


def test_get_github_summary_file_not_in_actions() -> None:
    """Test no summary file when not in GitHub Actions."""
    original = os.environ.get("GITHUB_ACTIONS")
    
    try:
        os.environ.pop("GITHUB_ACTIONS", None)
        opts = Options()
        result = get_github_summary_file(opts)
        assert result is None
    finally:
        if original is not None:
            os.environ["GITHUB_ACTIONS"] = original


def test_write_github_summary_success() -> None:
    """Test writing GitHub summary for successful build."""
    with TemporaryDirectory() as d:
        summary_file = Path(d) / "summary.md"
        opts = Options(flake_url=".#checks", flake_fragment="checks")
        
        results = [
            Result(
                result_type=ResultType.EVAL,
                attr="test-package",
                success=True,
                duration=1.5,
                error=None,
            ),
            Result(
                result_type=ResultType.BUILD,
                attr="test-package",
                success=True,
                duration=10.2,
                error=None,
            ),
        ]
        
        write_github_summary(summary_file, opts, results, rc=0)
        
        content = summary_file.read_text()
        assert "# nix-fast-build Results" in content
        assert "✅ Build Successful" in content
        assert "EVAL" in content
        assert "BUILD" in content


def test_write_github_summary_failure() -> None:
    """Test writing GitHub summary for failed build with logs."""
    with TemporaryDirectory() as d:
        summary_file = Path(d) / "summary.md"
        opts = Options(flake_url=".#checks", flake_fragment="checks")
        
        results = [
            Result(
                result_type=ResultType.EVAL,
                attr="test-package",
                success=True,
                duration=1.5,
                error=None,
            ),
            Result(
                result_type=ResultType.BUILD,
                attr="test-package",
                success=False,
                duration=5.3,
                error="build exited with 1",
                log_output="error: builder failed\nsome build output\n",
            ),
        ]
        
        write_github_summary(summary_file, opts, results, rc=1)
        
        content = summary_file.read_text()
        assert "# nix-fast-build Results" in content
        assert "❌ Build Failed" in content
        assert "## Failed Builds" in content
        assert "test-package" in content
        assert "build exited with 1" in content
        assert "error: builder failed" in content
        assert "<details>" in content
        assert "Build Log" in content
