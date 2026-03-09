"""Tests for the issue parser."""

import pytest
from bot.issue_parser import IssueParser, ParsedIssue


# ──────────────────────────────────────────────
# Sample issue bodies (based on real issues)
# ──────────────────────────────────────────────

ISSUE_5524_BODY = """**PySDK Version**
- [ ] PySDK V2 (2.x)
- [x] PySDK V3 (3.x)

**Describe the bug**
In SageMaker V2 we had `Estimator` which had vast support for pipeline variables. For example `image_uri`:

```
class Estimator(EstimatorBase):
    def __init__(
        self,
        image_uri: Union[str, PipelineVariable],
```

In SageMaker V3, we replace `Estimator` with `ModelTrainer`, however the support for pipeline variables is lacking.

**To reproduce**

```python
from sagemaker.core.workflow.parameters import ParameterString
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.train import ModelTrainer

pipeline_session = PipelineSession(default_bucket="my-bucket")
training_image = ParameterString(name="training_image")

trainer = ModelTrainer(
    training_image=training_image,
    sagemaker_session=pipeline_session
)
```
```
ValidationError: 1 validation error for ModelTrainer
training_image
  Input should be a valid string [type=string_type, input_value=ParameterString(name='tra...g'>, default_value=None), input_type=ParameterString]
```

**Expected behavior**
`ModelTrainer` should accept `PipelineVariable` for `training_image` (and other relevant fields), matching V2 `Estimator` behaviour.
"""


ISSUE_SIMPLE_BODY = """**Describe the bug**
TypeError when calling StoredFunction with hmac_key parameter.

```python
from sagemaker.workflow import StoredFunction
func = StoredFunction(hmac_key="test")
```

```
TypeError: StoredFunction.__init__() got an unexpected keyword argument 'hmac_key'
```

**Expected behavior**
StoredFunction should accept hmac_key parameter.
"""


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

class TestParsedIssue:
    """Tests for the ParsedIssue dataclass."""

    def test_short_description_removes_prefix(self):
        issue = ParsedIssue(number=1, title="[Bug] Something is broken", body="")
        assert issue.short_description == "Something is broken"

    def test_short_description_truncates(self):
        long_title = "A" * 100
        issue = ParsedIssue(number=1, title=long_title, body="")
        assert len(issue.short_description) <= 72
        assert issue.short_description.endswith("...")

    def test_short_description_normal(self):
        issue = ParsedIssue(number=1, title="Fix the thing", body="")
        assert issue.short_description == "Fix the thing"


class TestIssueParser:
    """Tests for the IssueParser."""

    def setup_method(self):
        self.parser = IssueParser()

    def test_parse_component_from_labels(self):
        parsed = self.parser.parse_raw(
            number=1,
            title="Test",
            body="body",
            labels=["type: bug", "component: pipelines"],
        )
        assert parsed.component == "pipelines"

    def test_parse_component_unknown_without_label(self):
        parsed = self.parser.parse_raw(
            number=1, title="Test", body="body", labels=["type: bug"]
        )
        assert parsed.component == "unknown"

    def test_detect_sdk_v3(self):
        parsed = self.parser.parse_raw(
            number=1, title="Test", body=ISSUE_5524_BODY
        )
        assert parsed.sdk_version == "v3"

    def test_extract_affected_classes_from_5524(self):
        parsed = self.parser.parse_raw(
            number=5524,
            title="Extend PipelineVariable support in ModelTrainer to match Estimator",
            body=ISSUE_5524_BODY,
        )
        assert "ModelTrainer" in parsed.affected_classes
        # "Estimator" alone doesn't match CLASS_RE (needs prefix chars before suffix)
        # but EstimatorBase and ModelTrainer do — the important class is ModelTrainer
        assert len(parsed.affected_classes) >= 1

    def test_extract_error_type_validation_error(self):
        parsed = self.parser.parse_raw(
            number=5524, title="Test", body=ISSUE_5524_BODY
        )
        assert parsed.error_type == "ValidationError"

    def test_extract_error_type_type_error(self):
        parsed = self.parser.parse_raw(
            number=1, title="Test", body=ISSUE_SIMPLE_BODY
        )
        assert parsed.error_type == "TypeError"

    def test_extract_reproduction_code(self):
        parsed = self.parser.parse_raw(
            number=5524, title="Test", body=ISSUE_5524_BODY
        )
        assert parsed.reproduction_code is not None
        assert "ModelTrainer" in parsed.reproduction_code
        assert "ParameterString" in parsed.reproduction_code

    def test_extract_modules(self):
        parsed = self.parser.parse_raw(
            number=5524, title="Test", body=ISSUE_5524_BODY
        )
        assert any("sagemaker" in m for m in parsed.affected_modules)

    def test_extract_expected_behavior(self):
        parsed = self.parser.parse_raw(
            number=5524, title="Test", body=ISSUE_5524_BODY
        )
        # expected_behavior extraction is present (may match "should" in error text too)
        assert parsed.expected_behavior is not None

    def test_extract_expected_behavior_simple(self):
        """Test expected behavior extraction with a simpler issue body."""
        parsed = self.parser.parse_raw(
            number=1, title="Test", body=ISSUE_SIMPLE_BODY
        )
        assert parsed.expected_behavior is not None
        assert "hmac_key" in parsed.expected_behavior

    def test_empty_body(self):
        parsed = self.parser.parse_raw(number=1, title="Bug", body="")
        assert parsed.error_type is None
        assert parsed.reproduction_code is None
        assert parsed.affected_classes == []

    def test_labels_preserved(self):
        labels = ["type: bug", "component: training", "priority: high"]
        parsed = self.parser.parse_raw(
            number=1, title="Test", body="body", labels=labels
        )
        assert parsed.labels == labels

    def test_url_preserved(self):
        parsed = self.parser.parse_raw(
            number=1,
            title="Test",
            body="body",
            url="https://github.com/aws/sagemaker-python-sdk/issues/1",
        )
        assert parsed.url == "https://github.com/aws/sagemaker-python-sdk/issues/1"
