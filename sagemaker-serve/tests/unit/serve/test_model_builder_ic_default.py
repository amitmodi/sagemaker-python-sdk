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
"""Unit tests for IC-default integration in ModelBuilder.deploy().

Tests the _try_derive_ic_resource_requirements() helper method and the
deploy() method's IC-default behavior.
"""
from __future__ import annotations

import os
from enum import Enum
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from sagemaker.core.inference_config import ResourceRequirements
from sagemaker.core.enums import EndpointType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model_builder(**overrides):
    """Create a minimal mock ModelBuilder with IC-default-relevant attributes."""
    from sagemaker.serve.model_builder import ModelBuilder

    mb = MagicMock(spec=ModelBuilder)

    # Set default attributes that _try_derive_ic_resource_requirements uses
    mb.model = overrides.get("model", "some-model-id")
    mb.model_server = overrides.get("model_server", "TGI")
    mb.instance_type = overrides.get("instance_type", "ml.g5.xlarge")
    mb.region = overrides.get("region", "us-east-1")
    mb.sagemaker_session = overrides.get("sagemaker_session", MagicMock())

    # Bind the real method to our mock
    from sagemaker.serve.model_builder import ModelBuilder as RealMB

    mb._try_derive_ic_resource_requirements = RealMB._try_derive_ic_resource_requirements.__get__(
        mb, type(mb)
    )

    return mb


# ===========================================================================
# Tests for _try_derive_ic_resource_requirements
# ===========================================================================


class TestTryDeriveICResourceRequirements:
    """Tests for the _try_derive_ic_resource_requirements helper method."""

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_returns_resource_requirements_on_success(self, mock_derive):
        """When derivation succeeds, should return ResourceRequirements."""
        expected_rr = ResourceRequirements(
            requests={"memory": 24576, "num_cpus": 4, "num_accelerators": 1, "copies": 1}
        )
        mock_derive.return_value = expected_rr

        mb = _make_mock_model_builder()
        result = mb._try_derive_ic_resource_requirements(initial_instance_count=1)

        assert result is expected_rr
        mock_derive.assert_called_once()

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_returns_none_when_derive_returns_none(self, mock_derive):
        """When derive_resource_requirements returns None (gate 1 opt-out), should return None."""
        mock_derive.return_value = None

        mb = _make_mock_model_builder()
        result = mb._try_derive_ic_resource_requirements(initial_instance_count=1)

        assert result is None

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_returns_none_on_value_error(self, mock_derive):
        """When derive raises ValueError (gates 2-4, 8), should catch and return None."""
        mock_derive.side_effect = ValueError("Instance type not supported")

        mb = _make_mock_model_builder()
        result = mb._try_derive_ic_resource_requirements(initial_instance_count=1)

        assert result is None

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_returns_none_on_invalid_min_memory(self, mock_derive):
        """When derived RR has None/zero min_memory, should return None."""
        bad_rr = ResourceRequirements(requests={"memory": 0, "num_cpus": 4, "copies": 1})
        mock_derive.return_value = bad_rr

        mb = _make_mock_model_builder()
        result = mb._try_derive_ic_resource_requirements(initial_instance_count=1)

        assert result is None

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_passes_endpoint_type_to_derive(self, mock_derive):
        """Should pass the endpoint_type parameter through to derive_resource_requirements."""
        mock_derive.return_value = None

        mb = _make_mock_model_builder()
        mb._try_derive_ic_resource_requirements(
            initial_instance_count=1, endpoint_type=EndpointType.MODEL_BASED
        )

        call_kwargs = mock_derive.call_args[1]
        assert call_kwargs["endpoint_type"] == EndpointType.MODEL_BASED

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_endpoint_type_none_passed_correctly(self, mock_derive):
        """When endpoint_type is None (default), should pass None to derive."""
        mock_derive.return_value = None

        mb = _make_mock_model_builder()
        mb._try_derive_ic_resource_requirements(initial_instance_count=1, endpoint_type=None)

        call_kwargs = mock_derive.call_args[1]
        assert call_kwargs["endpoint_type"] is None

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_model_server_none_handled(self, mock_derive):
        """When model_server is None, should pass None (not 'None') to derive."""
        mock_derive.return_value = None

        mb = _make_mock_model_builder(model_server=None)
        mb._try_derive_ic_resource_requirements(initial_instance_count=1)

        call_kwargs = mock_derive.call_args[1]
        assert call_kwargs["model_server"] is None

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_model_server_string_passed(self, mock_derive):
        """When model_server is set, should be converted to string."""
        mock_derive.return_value = None

        mb = _make_mock_model_builder(model_server="DJL_SERVING")
        mb._try_derive_ic_resource_requirements(initial_instance_count=1)

        call_kwargs = mock_derive.call_args[1]
        assert call_kwargs["model_server"] == "DJL_SERVING"

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_model_id_set_for_string_model(self, mock_derive):
        """When model is a string (JumpStart ID), should pass it as model_id."""
        mock_derive.return_value = None

        mb = _make_mock_model_builder(model="huggingface-llm-falcon-7b-bf16")
        mb._try_derive_ic_resource_requirements(initial_instance_count=1)

        call_kwargs = mock_derive.call_args[1]
        assert call_kwargs["model_id"] == "huggingface-llm-falcon-7b-bf16"

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_model_id_none_for_object_model(self, mock_derive):
        """When model is not a string, model_id should be None."""
        mock_derive.return_value = None

        mb = _make_mock_model_builder(model=MagicMock())  # Non-string model object
        mb._try_derive_ic_resource_requirements(initial_instance_count=1)

        call_kwargs = mock_derive.call_args[1]
        assert call_kwargs["model_id"] is None

    @patch("sagemaker.serve.utils.ic_utils.derive_resource_requirements")
    def test_returns_none_on_negative_min_memory(self, mock_derive):
        """Negative min_memory should be caught by post-derivation validation."""
        bad_rr = ResourceRequirements(requests={"memory": -100, "num_cpus": 4, "copies": 1})
        mock_derive.return_value = bad_rr

        mb = _make_mock_model_builder()
        result = mb._try_derive_ic_resource_requirements(initial_instance_count=1)

        assert result is None


# ===========================================================================
# Tests for ResourceRequirements API compatibility
# ===========================================================================


class TestResourceRequirementsCompatibility:
    """Verify ResourceRequirements created by ic_utils works with _deploy_core_endpoint."""

    def test_get_compute_resource_requirements_gpu(self):
        """GPU ResourceRequirements should produce correct dict."""
        rr = ResourceRequirements(
            requests={
                "num_accelerators": 1,
                "memory": 24576,
                "num_cpus": 4,
                "copies": 1,
            }
        )
        compute_rr = rr.get_compute_resource_requirements()
        assert compute_rr["MinMemoryRequiredInMb"] == 24576
        assert compute_rr["NumberOfCpuCoresRequired"] == 4
        assert compute_rr["NumberOfAcceleratorDevicesRequired"] == 1
        assert rr.copy_count == 1

    def test_get_compute_resource_requirements_cpu(self):
        """CPU ResourceRequirements should NOT have accelerator key."""
        rr = ResourceRequirements(
            requests={
                "memory": 16384,
                "num_cpus": 4,
                "copies": 1,
            }
        )
        compute_rr = rr.get_compute_resource_requirements()
        assert compute_rr["MinMemoryRequiredInMb"] == 16384
        assert compute_rr["NumberOfCpuCoresRequired"] == 4
        assert "NumberOfAcceleratorDevicesRequired" not in compute_rr
        assert rr.copy_count == 1

    def test_scale_to_zero_requirements(self):
        """Scale-to-zero should have copies=0."""
        rr = ResourceRequirements(
            requests={"memory": 1024, "copies": 0}
        )
        assert rr.copy_count == 0
        assert rr.min_memory == 1024
