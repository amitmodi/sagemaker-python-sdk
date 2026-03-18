# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Unit tests for sagemaker.serve.utils.ic_utils — the 8-gate algorithm."""
from __future__ import annotations

import os
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.serve.utils.ic_utils import (
    IC_DEFAULT_COPIES,
    IC_MIN_MEMORY_MB,
    IC_MIN_VCPUS,
    IC_UNSUPPORTED_INSTANCE_FAMILIES,
    IC_UNSUPPORTED_REGIONS,
    INSTANCE_SPEC_REGISTRY,
    _assert_ic_compatible,
    _derive_from_instance_spec,
    _is_ic_default_disabled,
    _is_ic_unsupported_instance,
    _validate_resource_requirements,
    derive_resource_requirements,
    region_supports_inference_components,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

class _MockEndpointType(Enum):
    MODEL_BASED = "ModelBased"
    INFERENCE_COMPONENT_BASED = "InferenceComponentBased"


@pytest.fixture
def mock_endpoint_type_model_based():
    """Return a mock EndpointType.MODEL_BASED that matches the real enum."""
    with patch("sagemaker.serve.utils.ic_utils.EndpointType", _MockEndpointType):
        # Need to also patch at import location
        yield _MockEndpointType.MODEL_BASED


@pytest.fixture
def mock_endpoint_type_ic():
    yield _MockEndpointType.INFERENCE_COMPONENT_BASED


def _make_resource_requirements(**kwargs):
    """Create a mock ResourceRequirements with specified attributes."""
    rr = MagicMock()
    rr.min_memory = kwargs.get("min_memory", 2048)
    rr.num_cpus = kwargs.get("num_cpus", 2)
    rr.num_accelerators = kwargs.get("num_accelerators", None)
    rr.copy_count = kwargs.get("copy_count", 1)
    rr.max_memory = kwargs.get("max_memory", None)
    return rr


# ===========================================================================
# Gate 2 Tests: _assert_ic_compatible
# ===========================================================================


class TestAssertICCompatible:
    """Tests for gate 2 — model/server IC compatibility checks."""

    def test_pipeline_model_raises(self):
        """Pipeline models (list) should raise ValueError."""
        with pytest.raises(ValueError, match="PipelineModel"):
            _assert_ic_compatible(model_server=None, model=["model_a", "model_b"])

    def test_triton_server_raises(self):
        """Triton model server should raise ValueError."""
        with pytest.raises(ValueError, match="Triton"):
            _assert_ic_compatible(model_server="TRITON", model="some_model")

    def test_triton_case_insensitive(self):
        """Triton detection should be case-insensitive."""
        with pytest.raises(ValueError, match="Triton"):
            _assert_ic_compatible(model_server="triton", model="some_model")

    def test_torchserve_passes(self):
        """TorchServe (single model) should pass without error."""
        # Should not raise — single model TorchServe is IC-compatible
        _assert_ic_compatible(model_server="TORCHSERVE", model="some_model")

    def test_none_server_passes(self):
        """None model server should pass."""
        _assert_ic_compatible(model_server=None, model="some_model")

    def test_djl_serving_passes(self):
        """DJL Serving should pass."""
        _assert_ic_compatible(model_server="DJL_SERVING", model="some_model")

    def test_tgi_passes(self):
        """TGI should pass."""
        _assert_ic_compatible(model_server="TGI", model="some_model")

    def test_empty_list_raises(self):
        """Empty list model should raise (still a pipeline model)."""
        with pytest.raises(ValueError, match="PipelineModel"):
            _assert_ic_compatible(model_server=None, model=[])


# ===========================================================================
# Gate 3 Tests: _is_ic_unsupported_instance
# ===========================================================================


class TestIsICUnsupportedInstance:
    """Tests for gate 3 — unsupported instance type detection."""

    @pytest.mark.parametrize(
        "instance_type",
        [
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5d.large",
            "ml.c5.xlarge",
            "ml.c5d.2xlarge",
            "ml.p2.xlarge",
            "ml.p2.8xlarge",
        ],
    )
    def test_unsupported_families(self, instance_type):
        """Known unsupported instance families should return True."""
        assert _is_ic_unsupported_instance(instance_type) is True

    @pytest.mark.parametrize(
        "instance_type",
        [
            "ml.g5.xlarge",
            "ml.g5.12xlarge",
            "ml.m6i.xlarge",
            "ml.c6i.large",
            "ml.p3.2xlarge",
            "ml.p4d.24xlarge",
            "ml.inf2.xlarge",
        ],
    )
    def test_supported_families(self, instance_type):
        """Supported instance families should return False."""
        assert _is_ic_unsupported_instance(instance_type) is False

    def test_empty_string(self):
        """Empty string should return False (not unsupported)."""
        assert _is_ic_unsupported_instance("") is False

    def test_none_string(self):
        """None should return False."""
        assert _is_ic_unsupported_instance(None) is False

    def test_malformed_instance_type(self):
        """Malformed instance type should return False."""
        assert _is_ic_unsupported_instance("invalid") is False


# ===========================================================================
# Gate 4 Tests: region_supports_inference_components
# ===========================================================================


class TestRegionSupportsIC:
    """Tests for gate 4 — region support checks."""

    @pytest.mark.parametrize(
        "region",
        [
            "us-east-1",
            "us-west-2",
            "eu-west-1",
            "ap-northeast-1",
            "ap-southeast-1",
        ],
    )
    def test_supported_regions(self, region):
        assert region_supports_inference_components(region) is True

    @pytest.mark.parametrize("region", sorted(IC_UNSUPPORTED_REGIONS))
    def test_unsupported_regions(self, region):
        assert region_supports_inference_components(region) is False

    def test_none_region(self):
        assert region_supports_inference_components(None) is False

    def test_empty_region(self):
        assert region_supports_inference_components("") is False


# ===========================================================================
# Gate 5 Tests: _validate_resource_requirements
# ===========================================================================


class TestValidateResourceRequirements:
    """Tests for gate 5 — resource requirement minimums."""

    def test_valid_requirements_pass(self):
        """Requirements above minimums should pass."""
        rr = _make_resource_requirements(min_memory=2048, num_cpus=2)
        _validate_resource_requirements(rr)  # Should not raise

    def test_memory_below_minimum_raises(self):
        """Memory below 1024 MB should raise ValueError."""
        rr = _make_resource_requirements(min_memory=512)
        with pytest.raises(ValueError, match="min_memory"):
            _validate_resource_requirements(rr)

    def test_cpus_below_minimum_raises(self):
        """vCPUs below 1 should raise ValueError."""
        rr = _make_resource_requirements(num_cpus=0)
        with pytest.raises(ValueError, match="num_cpus"):
            _validate_resource_requirements(rr)

    def test_none_memory_passes(self):
        """None memory should pass (not yet specified)."""
        rr = _make_resource_requirements(min_memory=None)
        _validate_resource_requirements(rr)

    def test_none_cpus_passes(self):
        """None cpus should pass (not yet specified)."""
        rr = _make_resource_requirements(num_cpus=None)
        _validate_resource_requirements(rr)

    def test_exact_minimum_passes(self):
        """Exact minimums should pass."""
        rr = _make_resource_requirements(min_memory=1024, num_cpus=1)
        _validate_resource_requirements(rr)


# ===========================================================================
# Gate 8 Tests: _derive_from_instance_spec
# ===========================================================================


class TestDeriveFromInstanceSpec:
    """Tests for gate 8 — instance spec registry derivation."""

    def test_gpu_instance_g5_xlarge(self):
        """G5.xlarge should derive GPU requirements correctly."""
        rr = _derive_from_instance_spec("ml.g5.xlarge")
        assert rr is not None
        assert rr.num_accelerators == 1
        # memory = per_gpu_memory_mb * num_accelerators = 24576 * 1
        assert rr.min_memory == 24576
        assert rr.copy_count == IC_DEFAULT_COPIES

    def test_gpu_instance_g5_12xlarge(self):
        """G5.12xlarge (4 GPUs) should derive multi-GPU requirements."""
        rr = _derive_from_instance_spec("ml.g5.12xlarge")
        assert rr is not None
        assert rr.num_accelerators == 4
        assert rr.min_memory == 24576 * 4  # 98304
        assert rr.copy_count == IC_DEFAULT_COPIES

    def test_gpu_instance_p4d(self):
        """P4d.24xlarge should derive 8-GPU requirements."""
        rr = _derive_from_instance_spec("ml.p4d.24xlarge")
        assert rr is not None
        assert rr.num_accelerators == 8
        assert rr.min_memory == 40960 * 8  # 327680
        assert rr.copy_count == IC_DEFAULT_COPIES

    def test_cpu_instance_m6i(self):
        """M6i.xlarge should derive CPU-only requirements (no accelerators)."""
        rr = _derive_from_instance_spec("ml.m6i.xlarge")
        assert rr is not None
        assert rr.num_accelerators is None
        assert rr.min_memory == 16384
        assert rr.num_cpus == 4
        assert rr.copy_count == IC_DEFAULT_COPIES

    def test_cpu_instance_c6i(self):
        """C6i compute-optimized should return CPU requirements."""
        rr = _derive_from_instance_spec("ml.c6i.xlarge")
        assert rr is not None
        assert rr.num_accelerators is None
        assert rr.min_memory == 8192
        assert rr.num_cpus == 4

    def test_inf2_instance(self):
        """Inf2 (Inferentia) should be treated like GPU instance."""
        rr = _derive_from_instance_spec("ml.inf2.xlarge")
        assert rr is not None
        assert rr.num_accelerators == 1

    def test_unknown_instance_returns_none(self):
        """Unknown instance type should return None."""
        rr = _derive_from_instance_spec("ml.x99.mega")
        assert rr is None

    def test_all_gpu_instances_have_accelerators(self):
        """Every GPU instance in registry should produce non-None accelerators."""
        for itype, spec in INSTANCE_SPEC_REGISTRY.items():
            if "num_accelerators" in spec:
                rr = _derive_from_instance_spec(itype)
                assert rr is not None
                assert rr.num_accelerators is not None
                assert rr.num_accelerators > 0

    def test_all_cpu_instances_have_no_accelerators(self):
        """Every CPU instance in registry should produce None accelerators."""
        for itype, spec in INSTANCE_SPEC_REGISTRY.items():
            if "num_accelerators" not in spec:
                rr = _derive_from_instance_spec(itype)
                assert rr is not None
                assert rr.num_accelerators is None

    def test_copies_always_one(self):
        """All derived requirements should have copies=1."""
        for itype in INSTANCE_SPEC_REGISTRY:
            rr = _derive_from_instance_spec(itype)
            assert rr.copy_count == 1


# ===========================================================================
# Environment/config opt-out tests
# ===========================================================================


class TestIsICDefaultDisabled:
    """Tests for environment/config opt-out logic."""

    def test_env_false_disables(self):
        with patch.dict(os.environ, {"SAGEMAKER_IC_DEFAULT": "false"}):
            assert _is_ic_default_disabled() is True

    def test_env_0_disables(self):
        with patch.dict(os.environ, {"SAGEMAKER_IC_DEFAULT": "0"}):
            assert _is_ic_default_disabled() is True

    def test_env_no_disables(self):
        with patch.dict(os.environ, {"SAGEMAKER_IC_DEFAULT": "no"}):
            assert _is_ic_default_disabled() is True

    def test_env_true_does_not_disable(self):
        with patch.dict(os.environ, {"SAGEMAKER_IC_DEFAULT": "true"}):
            assert _is_ic_default_disabled() is False

    def test_env_unset_does_not_disable(self):
        with patch.dict(os.environ, {}, clear=True):
            # get_config_value is lazy-imported; patch at source module
            with patch(
                "sagemaker.core.common_utils.get_config_value",
                side_effect=Exception("no config"),
            ):
                assert _is_ic_default_disabled() is False

    def test_env_case_insensitive(self):
        with patch.dict(os.environ, {"SAGEMAKER_IC_DEFAULT": "FALSE"}):
            assert _is_ic_default_disabled() is True

    def test_env_whitespace_stripped(self):
        with patch.dict(os.environ, {"SAGEMAKER_IC_DEFAULT": "  false  "}):
            assert _is_ic_default_disabled() is True


# ===========================================================================
# Main derive_resource_requirements() — full 8-gate integration tests
# ===========================================================================


class TestDeriveResourceRequirements:
    """Integration tests for the full 8-gate algorithm."""

    def _call(self, **kwargs):
        """Helper to call derive_resource_requirements with sensible defaults."""
        defaults = {
            "endpoint_type": _MockEndpointType.INFERENCE_COMPONENT_BASED,
            "model": "some-model",
            "model_server": "TGI",
            "instance_type": "ml.g5.xlarge",
            "region": "us-east-1",
            "initial_instance_count": 1,
            "resource_requirements": None,
            "model_id": None,
            "sagemaker_session": None,
        }
        defaults.update(kwargs)
        with patch("sagemaker.core.enums.EndpointType", _MockEndpointType):
            with patch.dict(os.environ, {}, clear=True):
                with patch(
                    "sagemaker.core.common_utils.get_config_value",
                    side_effect=Exception("no config"),
                ):
                    return derive_resource_requirements(**defaults)

    # --- Gate 1: MODEL_BASED opt-out ---

    def test_gate1_model_based_returns_none(self):
        """Explicit MODEL_BASED should return None."""
        result = self._call(endpoint_type=_MockEndpointType.MODEL_BASED)
        assert result is None

    # --- Gate 2: IC compatibility ---

    def test_gate2_pipeline_model_raises(self):
        """Pipeline model should raise ValueError."""
        with pytest.raises(ValueError, match="PipelineModel"):
            self._call(model=["m1", "m2"])

    def test_gate2_triton_raises(self):
        """Triton server should raise ValueError."""
        with pytest.raises(ValueError, match="Triton"):
            self._call(model_server="TRITON")

    # --- Gate 3: Unsupported instances ---

    def test_gate3_unsupported_instance_raises(self):
        """Unsupported instance family should raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            self._call(instance_type="ml.m5.xlarge")

    def test_gate3_m5d_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            self._call(instance_type="ml.m5d.2xlarge")

    # --- Gate 4: Region support ---

    def test_gate4_unsupported_region_raises(self):
        """GovCloud should raise ValueError."""
        with pytest.raises(ValueError, match="does not support"):
            self._call(region="us-gov-west-1")

    def test_gate4_china_region_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            self._call(region="cn-north-1")

    # --- Gate 5: Caller-supplied requirements ---

    def test_gate5_valid_requirements_returned(self):
        """Valid caller-supplied requirements should be returned as-is."""
        rr = _make_resource_requirements(min_memory=4096, num_cpus=4)
        result = self._call(resource_requirements=rr)
        assert result is rr

    def test_gate5_invalid_memory_raises(self):
        """Requirements with too-low memory should raise."""
        rr = _make_resource_requirements(min_memory=256)
        with pytest.raises(ValueError, match="min_memory"):
            self._call(resource_requirements=rr)

    def test_gate5_invalid_cpus_raises(self):
        """Requirements with 0 CPUs should raise."""
        rr = _make_resource_requirements(num_cpus=0)
        with pytest.raises(ValueError, match="num_cpus"):
            self._call(resource_requirements=rr)

    # --- Gate 6: Scale-to-zero ---

    def test_gate6_zero_instances_returns_scale_to_zero(self):
        """initial_instance_count=0 should return scale-to-zero requirements."""
        result = self._call(initial_instance_count=0)
        assert result is not None
        assert result.copy_count == 0
        assert result.min_memory == IC_MIN_MEMORY_MB

    # --- Gate 7: JumpStart (mocked) ---

    def test_gate7_jumpstart_success_returns_requirements(self):
        """Successful JumpStart fetch should return those requirements."""
        js_rr = _make_resource_requirements(min_memory=8192, num_cpus=4, num_accelerators=1)
        with patch(
            "sagemaker.serve.utils.ic_utils._try_retrieve_jumpstart_resource_requirements",
            return_value=js_rr,
        ):
            result = self._call(model_id="huggingface-llm-mistral-7b")
            assert result is js_rr

    def test_gate7_jumpstart_failure_falls_through(self):
        """JumpStart failure should fall through to gate 8."""
        with patch(
            "sagemaker.serve.utils.ic_utils._try_retrieve_jumpstart_resource_requirements",
            return_value=None,
        ):
            result = self._call(model_id="some-model-id", instance_type="ml.g5.xlarge")
            # Should get gate 8 result (from INSTANCE_SPEC_REGISTRY)
            assert result is not None
            assert result.num_accelerators == 1

    # --- Gate 8: Instance spec derivation ---

    def test_gate8_gpu_instance(self):
        """GPU instance should derive from registry."""
        result = self._call(instance_type="ml.g5.12xlarge")
        assert result is not None
        assert result.num_accelerators == 4
        assert result.min_memory == 24576 * 4
        assert result.copy_count == 1

    def test_gate8_cpu_instance(self):
        """CPU instance should derive from registry."""
        result = self._call(model_server="MMS", instance_type="ml.m6i.xlarge")
        assert result is not None
        assert result.num_accelerators is None
        assert result.min_memory == 16384
        assert result.num_cpus == 4

    def test_gate8_unknown_instance_raises(self):
        """Unknown instance not in registry should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot auto-derive"):
            self._call(instance_type="ml.z99.mega")

    # --- Full happy path ---

    def test_full_happy_path_gpu(self):
        """Full path: IC endpoint + supported region + supported instance → auto-derived."""
        result = self._call(
            endpoint_type=_MockEndpointType.INFERENCE_COMPONENT_BASED,
            model="my-model",
            model_server="TGI",
            instance_type="ml.g5.2xlarge",
            region="us-west-2",
            initial_instance_count=2,
        )
        assert result is not None
        assert result.num_accelerators == 1
        assert result.min_memory == 24576
        assert result.copy_count == 1

    def test_full_happy_path_cpu(self):
        """Full path for CPU instance."""
        result = self._call(
            model_server="MMS",
            instance_type="ml.c6i.2xlarge",
            region="eu-west-1",
        )
        assert result is not None
        assert result.min_memory == 16384
        assert result.num_cpus == 8
        assert result.num_accelerators is None

    # --- Gate priority tests ---

    def test_gate1_overrides_all(self):
        """MODEL_BASED should return None even with invalid instance/region."""
        result = self._call(
            endpoint_type=_MockEndpointType.MODEL_BASED,
            instance_type="ml.m5.xlarge",
            region="us-gov-west-1",
        )
        assert result is None

    def test_gate5_overrides_gates_6_7_8(self):
        """Caller requirements should be used even if instance_count=0."""
        rr = _make_resource_requirements(min_memory=2048, num_cpus=2)
        result = self._call(
            resource_requirements=rr,
            initial_instance_count=0,
        )
        assert result is rr  # Gate 5 wins over gate 6


# ===========================================================================
# Registry completeness tests
# ===========================================================================


class TestInstanceSpecRegistryCompleteness:
    """Validate the INSTANCE_SPEC_REGISTRY data."""

    def test_all_entries_have_cpus_and_memory(self):
        for itype, spec in INSTANCE_SPEC_REGISTRY.items():
            assert "num_cpus" in spec, f"{itype} missing num_cpus"
            assert "memory_mb" in spec, f"{itype} missing memory_mb"
            assert spec["num_cpus"] > 0, f"{itype} has invalid num_cpus"
            assert spec["memory_mb"] > 0, f"{itype} has invalid memory_mb"

    def test_gpu_entries_have_accelerator_fields(self):
        gpu_families = {"g5", "g6", "p3", "p4d", "p4de", "p5", "inf2"}
        for itype, spec in INSTANCE_SPEC_REGISTRY.items():
            family = itype.split(".")[1]
            if family in gpu_families:
                assert "num_accelerators" in spec, f"{itype} missing num_accelerators"
                assert "per_gpu_memory_mb" in spec, f"{itype} missing per_gpu_memory_mb"

    def test_no_unsupported_families_in_registry(self):
        """Registry should not contain entries from IC_UNSUPPORTED_INSTANCE_FAMILIES."""
        for itype in INSTANCE_SPEC_REGISTRY:
            parts = itype.split(".")
            family = f"{parts[0]}.{parts[1]}"
            assert family not in IC_UNSUPPORTED_INSTANCE_FAMILIES, (
                f"{itype} is in INSTANCE_SPEC_REGISTRY but its family "
                f"{family} is in IC_UNSUPPORTED_INSTANCE_FAMILIES"
            )

    def test_registry_has_minimum_entries(self):
        """Registry should have a reasonable number of entries."""
        assert len(INSTANCE_SPEC_REGISTRY) >= 30
