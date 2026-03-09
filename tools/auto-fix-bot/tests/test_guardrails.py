"""Tests for the guardrails module."""

import pytest
from pathlib import Path

from bot.guardrails import Guardrails, GuardrailResult, ChangeStats


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    "max_files_changed": 5,
    "max_lines_changed": 500,
    "require_tests_pass": False,  # Don't actually run tests in unit tests
    "confidence_threshold": 0.7,
    "daily_rate_limit": 5,
    "allowed_components": ["pipelines", "training"],
    "excluded_labels": ["wontfix", "duplicate", "needs-triage"],
}


@pytest.fixture
def guardrails():
    return Guardrails(DEFAULT_CONFIG)


@pytest.fixture
def small_change():
    return ChangeStats(
        files_changed=2,
        lines_added=10,
        lines_deleted=5,
        files=[
            {"path": "src/model.py", "additions": 5, "deletions": 4},
            {"path": "tests/test_model.py", "additions": 5, "deletions": 1},
        ],
    )


@pytest.fixture
def large_change():
    return ChangeStats(
        files_changed=8,
        lines_added=400,
        lines_deleted=200,
        files=[{"path": f"file_{i}.py", "additions": 50, "deletions": 25} for i in range(8)],
    )


# ──────────────────────────────────────────────
# Tests: Component check
# ──────────────────────────────────────────────

class TestComponentCheck:
    def test_allowed_component(self, guardrails):
        result = guardrails.check_component("pipelines")
        assert result.passed is True

    def test_blocked_component(self, guardrails):
        result = guardrails.check_component("unknown-thing")
        assert result.passed is False

    def test_no_allowlist(self):
        g = Guardrails({"allowed_components": []})
        result = g.check_component("anything")
        assert result.passed is True


# ──────────────────────────────────────────────
# Tests: Label check
# ──────────────────────────────────────────────

class TestLabelCheck:
    def test_no_excluded_labels(self, guardrails):
        result = guardrails.check_labels(["type: bug", "component: pipelines"])
        assert result.passed is True

    def test_excluded_label_blocks(self, guardrails):
        result = guardrails.check_labels(["type: bug", "wontfix"])
        assert result.passed is False

    def test_excluded_label_case_insensitive(self, guardrails):
        result = guardrails.check_labels(["type: bug", "Duplicate"])
        assert result.passed is False

    def test_empty_labels(self, guardrails):
        result = guardrails.check_labels([])
        assert result.passed is True


# ──────────────────────────────────────────────
# Tests: File/line limits
# ──────────────────────────────────────────────

class TestChangeLimits:
    def test_small_change_passes(self, guardrails, small_change):
        assert guardrails.check_files_changed(small_change).passed is True
        assert guardrails.check_lines_changed(small_change).passed is True

    def test_large_change_fails_files(self, guardrails, large_change):
        assert guardrails.check_files_changed(large_change).passed is False

    def test_large_change_fails_lines(self, guardrails, large_change):
        assert guardrails.check_lines_changed(large_change).passed is False

    def test_exact_limit_passes(self, guardrails):
        stats = ChangeStats(files_changed=5, lines_added=250, lines_deleted=250)
        assert guardrails.check_files_changed(stats).passed is True
        assert guardrails.check_lines_changed(stats).passed is True

    def test_one_over_limit_fails(self, guardrails):
        stats = ChangeStats(files_changed=6, lines_added=0, lines_deleted=0)
        assert guardrails.check_files_changed(stats).passed is False


# ──────────────────────────────────────────────
# Tests: Confidence / draft
# ──────────────────────────────────────────────

class TestConfidence:
    def test_confidence_always_passes(self, guardrails):
        # Confidence check never blocks — it just flags for draft
        assert guardrails.check_confidence(0.0).passed is True
        assert guardrails.check_confidence(1.0).passed is True

    def test_low_confidence_creates_draft(self, guardrails):
        assert guardrails.should_create_draft(0.5) is True
        assert guardrails.should_create_draft(0.69) is True

    def test_high_confidence_no_draft(self, guardrails):
        assert guardrails.should_create_draft(0.7) is False
        assert guardrails.should_create_draft(0.9) is False


# ──────────────────────────────────────────────
# Tests: Combined checks
# ──────────────────────────────────────────────

class TestAllChecks:
    def test_all_pass(self, guardrails, small_change):
        results = guardrails.check_all(
            repo_path=Path("/tmp/fake"),
            change_stats=small_change,
            confidence=0.9,
            component="pipelines",
            labels=["type: bug"],
        )
        assert guardrails.all_passed(results)

    def test_blocked_by_label(self, guardrails, small_change):
        results = guardrails.check_all(
            repo_path=Path("/tmp/fake"),
            change_stats=small_change,
            confidence=0.9,
            component="pipelines",
            labels=["type: bug", "wontfix"],
        )
        assert not guardrails.all_passed(results)


# ──────────────────────────────────────────────
# Tests: ChangeStats
# ──────────────────────────────────────────────

class TestChangeStats:
    def test_total_lines(self):
        stats = ChangeStats(lines_added=10, lines_deleted=5)
        assert stats.total_lines_changed == 15

    def test_empty_stats(self):
        stats = ChangeStats()
        assert stats.files_changed == 0
        assert stats.total_lines_changed == 0
