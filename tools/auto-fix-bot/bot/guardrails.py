"""Safety guardrails for the auto-fix bot."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChangeStats:
    """Statistics about code changes made by the agent."""

    files_changed: int = 0
    lines_added: int = 0
    lines_deleted: int = 0
    files: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_lines_changed(self) -> int:
        return self.lines_added + self.lines_deleted


class Guardrails:
    """Enforces safety limits on the bot's actions."""

    def __init__(self, config: dict[str, Any]):
        """Initialize guardrails from config.

        Args:
            config: The guardrails section of config.yaml
        """
        self.max_files = config.get("max_files_changed", 5)
        self.max_lines = config.get("max_lines_changed", 500)
        self.require_tests = config.get("require_tests_pass", True)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.daily_limit = config.get("daily_rate_limit", 5)
        self.allowed_components = config.get("allowed_components", [])
        self.excluded_labels = config.get("excluded_labels", [])

    def check_all(
        self,
        repo_path: Path,
        change_stats: ChangeStats,
        confidence: float,
        component: str = "unknown",
        labels: list[str] | None = None,
    ) -> list[GuardrailResult]:
        """Run all guardrail checks.

        Returns:
            List of GuardrailResult for each check
        """
        labels = labels or []
        results = []

        results.append(self.check_component(component))
        results.append(self.check_labels(labels))
        results.append(self.check_files_changed(change_stats))
        results.append(self.check_lines_changed(change_stats))
        results.append(self.check_confidence(confidence))

        if self.require_tests:
            results.append(self.check_tests_pass(repo_path, change_stats))

        return results

    def all_passed(self, results: list[GuardrailResult]) -> bool:
        """Check if all guardrail results passed."""
        return all(r.passed for r in results)

    def should_create_draft(self, confidence: float) -> bool:
        """Determine if the PR should be created as a draft."""
        return confidence < self.confidence_threshold

    # ── Individual checks ──

    def check_component(self, component: str) -> GuardrailResult:
        """Check if the component is in the allowlist."""
        if not self.allowed_components:
            return GuardrailResult(True, "No component allowlist configured")

        passed = component in self.allowed_components
        return GuardrailResult(
            passed=passed,
            message=(
                f"Component '{component}' is allowed"
                if passed
                else f"Component '{component}' not in allowlist: {self.allowed_components}"
            ),
            details={"component": component, "allowed": self.allowed_components},
        )

    def check_labels(self, labels: list[str]) -> GuardrailResult:
        """Check that no excluded labels are present."""
        blocked = [l for l in labels if l.lower() in [e.lower() for e in self.excluded_labels]]
        passed = len(blocked) == 0
        return GuardrailResult(
            passed=passed,
            message=(
                "No excluded labels found"
                if passed
                else f"Excluded label(s) found: {blocked}"
            ),
            details={"blocked_labels": blocked},
        )

    def check_files_changed(self, stats: ChangeStats) -> GuardrailResult:
        """Check that the number of files changed is within limits."""
        passed = stats.files_changed <= self.max_files
        return GuardrailResult(
            passed=passed,
            message=(
                f"Files changed: {stats.files_changed} (limit: {self.max_files})"
            ),
            details={
                "files_changed": stats.files_changed,
                "limit": self.max_files,
            },
        )

    def check_lines_changed(self, stats: ChangeStats) -> GuardrailResult:
        """Check that total lines changed is within limits."""
        passed = stats.total_lines_changed <= self.max_lines
        return GuardrailResult(
            passed=passed,
            message=(
                f"Lines changed: {stats.total_lines_changed} "
                f"(+{stats.lines_added}, -{stats.lines_deleted}, limit: {self.max_lines})"
            ),
            details={
                "lines_added": stats.lines_added,
                "lines_deleted": stats.lines_deleted,
                "total": stats.total_lines_changed,
                "limit": self.max_lines,
            },
        )

    def check_confidence(self, confidence: float) -> GuardrailResult:
        """Check agent's confidence level (always passes, but flags low confidence)."""
        # This check always passes — low confidence leads to a draft PR, not blocking
        return GuardrailResult(
            passed=True,
            message=f"Agent confidence: {confidence:.2f} (threshold: {self.confidence_threshold})",
            details={
                "confidence": confidence,
                "threshold": self.confidence_threshold,
                "is_draft": confidence < self.confidence_threshold,
            },
        )

    def check_tests_pass(
        self, repo_path: Path, stats: ChangeStats
    ) -> GuardrailResult:
        """Run pytest on modified test files to verify they pass."""
        test_files = [
            f["path"]
            for f in stats.files
            if "test" in f.get("path", "").lower()
        ]

        if not test_files:
            return GuardrailResult(
                passed=False,
                message="No test files found in changes — tests are required",
                details={"test_files": []},
            )

        logger.info("Running tests: %s", test_files)

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "-xvs"] + test_files,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            passed = result.returncode == 0
            return GuardrailResult(
                passed=passed,
                message=(
                    f"Tests passed ({len(test_files)} file(s))"
                    if passed
                    else f"Tests FAILED: {result.stdout[-500:]}"
                ),
                details={
                    "test_files": test_files,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-2000:] if not passed else "",
                    "stderr": result.stderr[-2000:] if not passed else "",
                },
            )
        except subprocess.TimeoutExpired:
            return GuardrailResult(
                passed=False,
                message="Tests timed out after 5 minutes",
                details={"test_files": test_files},
            )
        except Exception as e:
            return GuardrailResult(
                passed=False,
                message=f"Error running tests: {e}",
                details={"error": str(e)},
            )


def compute_change_stats(repo_path: Path) -> ChangeStats:
    """Compute change statistics from git diff in the repo.

    Must be called after changes have been staged (git add).
    """
    try:
        # Get diff stats
        result = subprocess.run(
            ["git", "diff", "--cached", "--numstat"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

        stats = ChangeStats()
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                added = int(parts[0]) if parts[0] != "-" else 0
                deleted = int(parts[1]) if parts[1] != "-" else 0
                filepath = parts[2]
                stats.files_changed += 1
                stats.lines_added += added
                stats.lines_deleted += deleted
                stats.files.append(
                    {
                        "path": filepath,
                        "additions": added,
                        "deletions": deleted,
                    }
                )

        return stats

    except Exception as e:
        logger.error("Failed to compute change stats: %s", e)
        return ChangeStats()
