"""Tools that the AI agent can invoke during the fix loop.

These map to Claude's tool-use format and wrap filesystem/shell operations
scoped to the cloned repository.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Tool definitions (Claude tool-use schema format)
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file at the given path (relative to the repo root). "
            "Returns the full file content as a string. Use this to examine source code, "
            "tests, and configuration files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repo root (e.g., 'sagemaker-train/src/sagemaker/train/model_trainer.py')",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file at the given path (relative to repo root). "
            "Creates the file if it doesn't exist, overwrites if it does. "
            "Use this to make code fixes and create test files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repo root",
                },
                "content": {
                    "type": "string",
                    "description": "The complete file content to write",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for a regex pattern across files in a directory. "
            "Returns matching lines with file paths and line numbers. "
            "Use this to find existing patterns, class definitions, imports, and usages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for (e.g., 'StrPipeVar', 'class ModelTrainer')",
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in, relative to repo root (default: '.')",
                    "default": ".",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern for file names (default: '*.py')",
                    "default": "*.py",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files and directories at the given path. "
            "Use this to explore the repository structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to repo root (default: '.')",
                    "default": ".",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth (default: 2)",
                    "default": 2,
                },
            },
            "required": [],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command in the repo directory. "
            "Use this for running tests (pytest), checking imports, or other validation. "
            "Only safe, read-only commands and test commands are allowed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run (e.g., 'python -m pytest tests/unit/train/test_model_trainer.py -xvs')",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "report_result",
        "description": (
            "Report the final result of the bug fix attempt. "
            "Call this when you've completed the fix (or determined the bug can't be fixed). "
            "This ends the tool-use loop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the bug was successfully fixed",
                },
                "confidence": {
                    "type": "number",
                    "description": "Your confidence level (0.0 to 1.0) that this fix is correct",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was changed and why",
                },
                "files_changed": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of files that were modified or created",
                },
                "reason": {
                    "type": "string",
                    "description": "If not successful, explain why the fix couldn't be made",
                },
            },
            "required": ["success", "confidence", "summary"],
        },
    },
]


# ──────────────────────────────────────────────
# Tool executor — runs tools within the repo sandbox
# ──────────────────────────────────────────────

# Commands that are allowed to run
ALLOWED_COMMANDS = [
    "python",
    "pytest",
    "grep",
    "find",
    "cat",
    "head",
    "tail",
    "wc",
    "diff",
    "git",
]

# Commands that are explicitly blocked
BLOCKED_COMMANDS = [
    "rm -rf",
    "sudo",
    "curl",
    "wget",
    "ssh",
    "scp",
    "aws ",
    "docker",
]

# Protected paths that cannot be written to
PROTECTED_PATHS = {
    ".git",
    ".github",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "conftest.py",
    ".env",
    ".secrets",
}

# Env var prefixes that must NOT be passed to subprocesses
SENSITIVE_ENV_PREFIXES = [
    "AWS_",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "ANTHROPIC_API_KEY",
    "TOKEN",
    "SECRET",
    "API_KEY",
    "CREDENTIAL",
    "PASSWORD",
]


def _build_safe_env() -> dict[str, str]:
    """Build a sanitized environment dict, stripping secrets."""
    safe = {}
    for key, val in os.environ.items():
        if any(key.upper().startswith(p) or key.upper() == p for p in SENSITIVE_ENV_PREFIXES):
            continue
        safe[key] = val
    safe["PYTHONDONTWRITEBYTECODE"] = "1"
    return safe


class ToolExecutor:
    """Executes tools within the cloned repo sandbox."""

    def __init__(self, repo_path: Path):
        """Initialize with the repo root path.

        Args:
            repo_path: Absolute path to the cloned repository root
        """
        self.repo_path = repo_path
        self.files_written: list[str] = []

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            tool_input: Tool input parameters

        Returns:
            Dict with the tool result (always has "content" key)
        """
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if not handler:
                return {"content": f"Unknown tool: {tool_name}", "is_error": True}
            return handler(tool_input)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return {"content": f"Error executing {tool_name}: {e}", "is_error": True}

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a relative path to an absolute path within the repo.

        Raises ValueError if the path would escape the repo.
        """
        resolved = (self.repo_path / relative_path).resolve()
        if not str(resolved).startswith(str(self.repo_path.resolve())):
            raise ValueError(f"Path escapes repo sandbox: {relative_path}")
        return resolved

    # ── Tool implementations ──

    def _tool_read_file(self, input: dict[str, Any]) -> dict[str, Any]:
        """Read a file's contents."""
        path = self._resolve_path(input["path"])
        if not path.exists():
            return {"content": f"File not found: {input['path']}", "is_error": True}
        if not path.is_file():
            return {"content": f"Not a file: {input['path']}", "is_error": True}

        content = path.read_text(encoding="utf-8")
        # Truncate very large files
        if len(content) > 50000:
            content = content[:50000] + "\n\n... [TRUNCATED — file too large] ..."
        return {"content": content}

    def _tool_write_file(self, input: dict[str, Any]) -> dict[str, Any]:
        """Write content to a file."""
        relative = input["path"]

        # Safety: block writes to protected paths (.git, setup.py, conftest.py, etc.)
        path_parts = Path(relative).parts
        for part in path_parts:
            if part in PROTECTED_PATHS:
                return {
                    "content": f"Write blocked: '{relative}' is a protected path ({part})",
                    "is_error": True,
                }
        # Also check the filename itself
        if Path(relative).name in PROTECTED_PATHS:
            return {
                "content": f"Write blocked: '{Path(relative).name}' is a protected file",
                "is_error": True,
            }

        # Safety: file size limit (1MB)
        if len(input["content"]) > 1_000_000:
            return {
                "content": "Write blocked: file content exceeds 1MB limit",
                "is_error": True,
            }

        path = self._resolve_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(input["content"], encoding="utf-8")
        self.files_written.append(input["path"])
        logger.info("Wrote file: %s (%d bytes)", input["path"], len(input["content"]))
        return {"content": f"Successfully wrote {input['path']}"}

    def _tool_search_files(self, input: dict[str, Any]) -> dict[str, Any]:
        """Search for a pattern across files."""
        pattern = input["pattern"]
        directory = input.get("directory", ".")
        file_pattern = input.get("file_pattern", "*.py")

        search_path = self._resolve_path(directory)
        if not search_path.exists():
            return {"content": f"Directory not found: {directory}", "is_error": True}

        result = subprocess.run(
            [
                "grep", "-rn", "--include", file_pattern,
                "-E", pattern,
                str(search_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = result.stdout
        if not output.strip():
            return {"content": f"No matches found for pattern: {pattern}"}

        # Make paths relative to repo root
        lines = output.strip().split("\n")
        relative_lines = []
        for line in lines[:100]:  # Limit to 100 matches
            try:
                line = line.replace(str(self.repo_path) + "/", "")
            except Exception:
                pass
            relative_lines.append(line)

        if len(lines) > 100:
            relative_lines.append(f"\n... and {len(lines) - 100} more matches")

        return {"content": "\n".join(relative_lines)}

    def _tool_list_files(self, input: dict[str, Any]) -> dict[str, Any]:
        """List files in a directory."""
        path = input.get("path", ".")
        max_depth = input.get("max_depth", 2)

        target = self._resolve_path(path)
        if not target.exists():
            return {"content": f"Path not found: {path}", "is_error": True}

        result = subprocess.run(
            ["find", str(target), "-maxdepth", str(max_depth), "-not", "-path", "*/.git/*"],
            capture_output=True,
            text=True,
            timeout=15,
        )

        lines = sorted(result.stdout.strip().split("\n"))
        relative_lines = []
        for line in lines:
            if line.strip():
                try:
                    relative_lines.append(str(Path(line).relative_to(self.repo_path)))
                except ValueError:
                    relative_lines.append(line)

        return {"content": "\n".join(relative_lines[:200])}

    def _tool_run_command(self, input: dict[str, Any]) -> dict[str, Any]:
        """Run a shell command (with safety checks)."""
        command = input["command"]

        # Safety check: block dangerous commands
        for blocked in BLOCKED_COMMANDS:
            if blocked in command:
                return {
                    "content": f"Command blocked for safety: contains '{blocked}'",
                    "is_error": True,
                }

        # Safety check: command must start with an allowed program
        first_word = command.split()[0] if command.split() else ""
        if first_word not in ALLOWED_COMMANDS:
            return {
                "content": (
                    f"Command not allowed: '{first_word}'. "
                    f"Allowed commands: {ALLOWED_COMMANDS}"
                ),
                "is_error": True,
            }

        # Safety check: block `python -c` (arbitrary code execution)
        # Only allow: python -m pytest, python --version, python --help
        if first_word == "python":
            args_preview = shlex.split(command)
            if len(args_preview) > 1 and args_preview[1] == "-c":
                return {
                    "content": "Command blocked: 'python -c' not allowed (arbitrary code execution risk). Use 'python -m pytest' instead.",
                    "is_error": True,
                }

        # Safety check: restrict git to read-only subcommands only
        # Block push, remote add, config --global, and other write operations
        if first_word == "git":
            args_preview = shlex.split(command)
            git_read_only = {"status", "diff", "log", "show", "branch", "tag", "ls-files", "rev-parse"}
            git_subcommand = args_preview[1] if len(args_preview) > 1 else ""
            if git_subcommand not in git_read_only:
                return {
                    "content": (
                        f"Command blocked: 'git {git_subcommand}' not allowed. "
                        f"Only read-only git commands are permitted: {sorted(git_read_only)}"
                    ),
                    "is_error": True,
                }

        logger.info("Running command: %s", command)

        try:
            # Use shlex.split() instead of shell=True to prevent shell injection
            args = shlex.split(command)
            result = subprocess.run(
                args,
                shell=False,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                env=_build_safe_env(),
            )

            output_parts = []
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout[-5000:]}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr[-2000:]}")

            output = "\n".join(output_parts)
            if not output:
                output = "(no output)"

            return {
                "content": f"Exit code: {result.returncode}\n{output}",
            }

        except subprocess.TimeoutExpired:
            return {"content": "Command timed out after 2 minutes", "is_error": True}

    def _tool_report_result(self, input: dict[str, Any]) -> dict[str, Any]:
        """Report the final result (captured by the agent loop)."""
        # This is a special tool — the agent loop handles it
        return {"content": "Result reported."}
