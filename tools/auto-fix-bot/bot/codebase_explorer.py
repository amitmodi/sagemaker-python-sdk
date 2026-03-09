"""Clone and explore the target repository."""

from __future__ import annotations

import logging
import subprocess
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CodebaseExplorer:
    """Clones the repo and provides exploration utilities."""

    def __init__(self, repo: str, workspace_dir: str = "workspace", branch: str = "master"):
        """Initialize the explorer.

        Args:
            repo: Repository in "owner/name" format
            workspace_dir: Local directory for cloning
            branch: Branch to clone
        """
        self.repo = repo
        self.repo_url = f"https://github.com/{repo}.git"
        self.workspace_dir = Path(workspace_dir)
        self.branch = branch
        self.repo_path: Path | None = None

    def clone(self) -> Path:
        """Clone the repository (shallow clone for speed).

        Returns:
            Path to the cloned repo root
        """
        repo_name = self.repo.split("/")[-1]
        self.repo_path = self.workspace_dir / repo_name

        if self.repo_path.exists():
            logger.info("Repo already cloned at %s, pulling latest", self.repo_path)
            subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=self.repo_path,
                capture_output=True,
            )
            return self.repo_path

        logger.info("Cloning %s (branch: %s) to %s", self.repo_url, self.branch, self.repo_path)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--branch", self.branch,
                self.repo_url,
                str(self.repo_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone repo: {result.stderr}")

        # Unshallow enough to create branches and diffs
        subprocess.run(
            ["git", "fetch", "--unshallow"],
            cwd=self.repo_path,
            capture_output=True,
        )

        logger.info("Cloned successfully to %s", self.repo_path)
        return self.repo_path

    def create_branch(self, branch_name: str) -> None:
        """Create and checkout a new branch.

        Args:
            branch_name: Name of the branch to create
        """
        if not self.repo_path:
            raise RuntimeError("Must clone repo first")

        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info("Created branch: %s", branch_name)

    def find_relevant_files(self, classes: list[str], modules: list[str]) -> dict[str, list[str]]:
        """Find source and test files relevant to the bug.

        Args:
            classes: Class names mentioned in the issue (e.g., ["ModelTrainer"])
            modules: Module paths mentioned (e.g., ["sagemaker.train"])

        Returns:
            Dict with "source_files" and "test_files" lists
        """
        if not self.repo_path:
            raise RuntimeError("Must clone repo first")

        source_files = set()
        test_files = set()

        # Search for class definitions
        for cls in classes:
            result = subprocess.run(
                ["grep", "-rl", f"class {cls}", "--include=*.py", "."],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            for path in result.stdout.strip().split("\n"):
                if path.strip():
                    path = path.lstrip("./")
                    if "test" in path.lower():
                        test_files.add(path)
                    else:
                        source_files.add(path)

        # Search for module imports
        for mod in modules:
            # Convert module path to file path pattern
            file_pattern = mod.replace(".", "/")
            result = subprocess.run(
                ["find", ".", "-path", f"*{file_pattern}*", "-name", "*.py"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            for path in result.stdout.strip().split("\n"):
                if path.strip():
                    path = path.lstrip("./")
                    if "test" in path.lower():
                        test_files.add(path)
                    else:
                        source_files.add(path)

        return {
            "source_files": sorted(source_files),
            "test_files": sorted(test_files),
        }

    def get_directory_tree(self, path: str = ".", max_depth: int = 3) -> str:
        """Get a directory tree listing (for the agent's context).

        Args:
            path: Relative path within the repo
            max_depth: Maximum directory depth

        Returns:
            String representation of the directory tree
        """
        if not self.repo_path:
            raise RuntimeError("Must clone repo first")

        target = self.repo_path / path
        if not target.exists():
            return f"Path not found: {path}"

        result = subprocess.run(
            ["find", str(target), "-maxdepth", str(max_depth), "-type", "f", "-name", "*.py"],
            capture_output=True,
            text=True,
        )

        files = sorted(result.stdout.strip().split("\n"))
        # Make paths relative to repo root
        relative_files = []
        for f in files:
            if f.strip():
                try:
                    relative_files.append(str(Path(f).relative_to(self.repo_path)))
                except ValueError:
                    relative_files.append(f)

        return "\n".join(relative_files)

    def cleanup(self) -> None:
        """Remove the cloned repository."""
        if self.repo_path and self.repo_path.exists():
            logger.info("Cleaning up %s", self.repo_path)
            shutil.rmtree(self.repo_path)
