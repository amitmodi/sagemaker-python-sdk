"""Shared pytest configuration and fixtures for the bot test suite."""

import pytest
from pathlib import Path


@pytest.fixture
def sample_issue_body():
    """Return a realistic issue body based on sagemaker-python-sdk #5524."""
    return """**PySDK Version**
- [ ] PySDK V2 (2.x)
- [x] PySDK V3 (3.x)

**Describe the bug**
`ModelTrainer` rejects `PipelineVariable` objects with a `ValidationError`.

**To reproduce**

```python
from sagemaker.train import ModelTrainer
from sagemaker.core.workflow.parameters import ParameterString

trainer = ModelTrainer(
    training_image=ParameterString(name="img"),
)
```
```
ValidationError: 1 validation error for ModelTrainer
training_image
  Input should be a valid string [type=string_type]
```

**Expected behavior**
`ModelTrainer` should accept `PipelineVariable` for `training_image`.
"""


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a temporary directory simulating a cloned repository."""
    src = tmp_path / "src" / "sagemaker" / "train"
    src.mkdir(parents=True)
    (src / "model_trainer.py").write_text(
        "class ModelTrainer:\n    training_image: str = None\n"
    )
    (src / "__init__.py").write_text("")

    tests = tmp_path / "tests" / "unit"
    tests.mkdir(parents=True)
    (tests / "test_model_trainer.py").write_text(
        "def test_example():\n    assert True\n"
    )

    return tmp_path
