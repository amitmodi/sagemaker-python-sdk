"""Validated configuration using Pydantic models.

Replaces raw dict access with type-safe, validated config objects.
Catches typos, missing fields, and wrong types at startup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class AIConfig(BaseModel):
    """AI model configuration."""

    provider: str = "anthropic"
    model: str = "claude-opus-4-20250514"
    max_tokens: int = Field(default=8192, ge=256, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_iterations: int = Field(default=25, ge=1, le=100)

    class BedrockConfig(BaseModel):
        region: str = "us-west-2"
        model_id: str = "anthropic.claude-sonnet-4-20250514-v1:0"

    bedrock: BedrockConfig = Field(default_factory=BedrockConfig)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in ("anthropic", "bedrock"):
            raise ValueError(f"provider must be 'anthropic' or 'bedrock', got '{v}'")
        return v


class GuardrailsConfig(BaseModel):
    """Safety guardrails configuration."""

    max_files_changed: int = Field(default=5, ge=1, le=50)
    max_lines_changed: int = Field(default=500, ge=1, le=5000)
    require_tests_pass: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    daily_rate_limit: int = Field(default=5, ge=1, le=100)
    allowed_components: list[str] = Field(default_factory=list)
    excluded_labels: list[str] = Field(
        default_factory=lambda: ["wontfix", "won't fix", "needs-triage", "duplicate"]
    )


class PRConfig(BaseModel):
    """PR creation configuration."""

    branch_template: str = "auto-fix/issue-{issue_number}"
    commit_template: str = "fix: {short_description} (fixes #{issue_number})"
    labels: list[str] = Field(default_factory=lambda: ["auto-generated"])
    draft_on_low_confidence: bool = True
    comment_on_issue: bool = True


class RepoConfig(BaseModel):
    """Target repository configuration."""

    owner: str = "aws"
    name: str = "sagemaker-python-sdk"
    default_branch: str = "master"

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    dir: str = "logs"
    save_conversation: bool = True

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"level must be one of {valid}, got '{v}'")
        return v.upper()


class WorkspaceConfig(BaseModel):
    """Workspace configuration."""

    dir: str = "workspace"
    cleanup: bool = False


class BotConfig(BaseModel):
    """Root configuration model — validates the entire config.yaml."""

    repo: RepoConfig = Field(default_factory=RepoConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    pr: PRConfig = Field(default_factory=PRConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)


def load_validated_config(config_path: str = "config/config.yaml") -> BotConfig:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Validated BotConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        pydantic.ValidationError: If config values are invalid
    """
    path = Path(config_path)
    if not path.exists():
        # Fall back to default config next to the bot package
        path = Path(__file__).parent.parent / "config" / "config.yaml"

    if not path.exists():
        # Use all defaults
        return BotConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return BotConfig(**raw)
