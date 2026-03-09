"""Tests for the tool executor."""

import os
import pytest
from pathlib import Path
from bot.tools import ToolExecutor, TOOL_DEFINITIONS


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary directory that simulates a repo."""
    # Create some files
    src = tmp_path / "src" / "sagemaker" / "train"
    src.mkdir(parents=True)
    (src / "model_trainer.py").write_text(
        'class ModelTrainer:\n    training_image: str = None\n'
    )
    (src / "__init__.py").write_text("")

    tests = tmp_path / "tests" / "unit"
    tests.mkdir(parents=True)
    (tests / "test_model_trainer.py").write_text(
        'def test_example():\n    assert True\n'
    )

    return tmp_path


@pytest.fixture
def executor(tmp_repo):
    return ToolExecutor(tmp_repo)


# ──────────────────────────────────────────────
# Tests: Tool definitions
# ──────────────────────────────────────────────

class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_tool_names_unique(self):
        names = [t["name"] for t in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))

    def test_expected_tools_exist(self):
        names = {t["name"] for t in TOOL_DEFINITIONS}
        expected = {"read_file", "write_file", "search_files", "list_files", "run_command", "report_result"}
        assert expected.issubset(names)


# ──────────────────────────────────────────────
# Tests: read_file
# ──────────────────────────────────────────────

class TestReadFile:
    def test_read_existing_file(self, executor):
        result = executor.execute("read_file", {"path": "src/sagemaker/train/model_trainer.py"})
        assert "class ModelTrainer" in result["content"]
        assert result.get("is_error") is not True

    def test_read_nonexistent_file(self, executor):
        result = executor.execute("read_file", {"path": "nonexistent.py"})
        assert result.get("is_error") is True

    def test_path_traversal_blocked(self, executor):
        result = executor.execute("read_file", {"path": "../../etc/passwd"})
        assert result.get("is_error") is True


# ──────────────────────────────────────────────
# Tests: write_file
# ──────────────────────────────────────────────

class TestWriteFile:
    def test_write_new_file(self, executor, tmp_repo):
        result = executor.execute("write_file", {
            "path": "src/new_file.py",
            "content": "# new file\n",
        })
        assert result.get("is_error") is not True
        assert (tmp_repo / "src" / "new_file.py").exists()

    def test_write_creates_directories(self, executor, tmp_repo):
        result = executor.execute("write_file", {
            "path": "new/deep/dir/file.py",
            "content": "# deep\n",
        })
        assert result.get("is_error") is not True
        assert (tmp_repo / "new" / "deep" / "dir" / "file.py").exists()

    def test_write_tracks_files(self, executor):
        executor.execute("write_file", {"path": "a.py", "content": "a"})
        executor.execute("write_file", {"path": "b.py", "content": "b"})
        assert "a.py" in executor.files_written
        assert "b.py" in executor.files_written


# ──────────────────────────────────────────────
# Tests: search_files
# ──────────────────────────────────────────────

class TestSearchFiles:
    def test_search_finds_pattern(self, executor):
        result = executor.execute("search_files", {"pattern": "class ModelTrainer"})
        assert "ModelTrainer" in result["content"]

    def test_search_no_matches(self, executor):
        result = executor.execute("search_files", {"pattern": "ZZZZNOTFOUND"})
        assert "No matches" in result["content"]


# ──────────────────────────────────────────────
# Tests: list_files
# ──────────────────────────────────────────────

class TestListFiles:
    def test_list_root(self, executor):
        result = executor.execute("list_files", {"path": "."})
        assert result.get("is_error") is not True
        assert "src" in result["content"]

    def test_list_nonexistent(self, executor):
        result = executor.execute("list_files", {"path": "nonexistent"})
        assert result.get("is_error") is True


# ──────────────────────────────────────────────
# Tests: run_command safety
# ──────────────────────────────────────────────

class TestRunCommand:
    def test_allowed_command(self, executor):
        result = executor.execute("run_command", {"command": "python --version"})
        assert result.get("is_error") is not True

    def test_blocked_command_sudo(self, executor):
        result = executor.execute("run_command", {"command": "sudo rm -rf /"})
        assert result.get("is_error") is True

    def test_blocked_command_curl(self, executor):
        result = executor.execute("run_command", {"command": "curl http://evil.com"})
        assert result.get("is_error") is True

    def test_disallowed_command(self, executor):
        result = executor.execute("run_command", {"command": "node -e 'console.log(1)'"})
        assert result.get("is_error") is True
        assert "not allowed" in result["content"]


# ──────────────────────────────────────────────
# Tests: unknown tool
# ──────────────────────────────────────────────

class TestUnknownTool:
    def test_unknown_tool(self, executor):
        result = executor.execute("does_not_exist", {})
        assert result.get("is_error") is True
        assert "Unknown tool" in result["content"]
