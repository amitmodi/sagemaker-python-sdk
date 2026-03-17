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
"""Integration tests for IC-default deployment behavior.

These tests deploy real SageMaker endpoints and verify that:
1. IC-default creates INFERENCE_COMPONENT_BASED endpoints by default
2. MODEL_BASED opt-out works correctly

Requirements:
- Valid AWS credentials with SageMaker permissions
- Costs ~$0.25-0.50 per test run (ml.g5.2xlarge for ~10-15 min)

Run with:
    python -m pytest sagemaker-serve/tests/integ/test_ic_default_integration.py -v -s
"""
from __future__ import absolute_import

import json
import uuid
import pytest
import logging

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.core.resources import EndpointConfig, Endpoint
from sagemaker.core.enums import EndpointType
from sagemaker.train.configs import Compute

logger = logging.getLogger(__name__)

# Configuration
MODEL_ID = "huggingface-llm-falcon-7b-bf16"
MODEL_NAME_PREFIX = "ic-default-test-model"
ENDPOINT_NAME_PREFIX = "ic-default-test-ep"


@pytest.mark.slow_test
def test_ic_default_creates_inference_component():
    """Integration test: Verify IC-default creates an INFERENCE_COMPONENT_BASED endpoint.

    1. Build JumpStart model
    2. Deploy WITHOUT specifying endpoint_type (should auto-derive IC)
    3. Verify InferenceComponent exists on the endpoint
    4. Invoke endpoint and verify prediction
    5. Cleanup
    """
    core_model = None
    core_endpoint = None
    unique_id = str(uuid.uuid4())[:8]
    endpoint_name = f"{ENDPOINT_NAME_PREFIX}-ic-{unique_id}"

    try:
        # Build
        logger.info("Building JumpStart model for IC-default test...")
        compute = Compute(instance_type="ml.g5.2xlarge")
        jumpstart_config = JumpStartConfig(model_id=MODEL_ID)
        model_builder = ModelBuilder.from_jumpstart_config(
            jumpstart_config=jumpstart_config, compute=compute
        )

        core_model = model_builder.build(model_name=f"{MODEL_NAME_PREFIX}-{unique_id}")
        logger.info(f"Model created: {core_model.model_name}")

        # Deploy WITHOUT endpoint_type — IC-default should kick in
        logger.info("Deploying without endpoint_type (IC-default should activate)...")
        core_endpoint = model_builder.deploy(endpoint_name=endpoint_name)
        logger.info(f"Endpoint created: {core_endpoint.endpoint_name}")

        # Verify InferenceComponent exists
        logger.info("Verifying InferenceComponent was created...")
        sagemaker_client = model_builder.sagemaker_session.sagemaker_client
        ic_response = sagemaker_client.list_inference_components(
            EndpointNameEquals=endpoint_name
        )
        inference_components = ic_response.get("InferenceComponents", [])

        assert len(inference_components) > 0, (
            f"Expected IC-based endpoint with InferenceComponents, but found none on "
            f"endpoint '{endpoint_name}'. IC-default may not be working."
        )
        logger.info(
            f"✅ InferenceComponent(s) found: {[ic['InferenceComponentName'] for ic in inference_components]}"
        )

        # Invoke endpoint
        logger.info("Invoking endpoint...")
        test_data = {"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}}
        result = core_endpoint.invoke(
            body=json.dumps(test_data), content_type="application/json"
        )
        prediction = json.loads(result.body.read().decode("utf-8"))
        logger.info(f"✅ Prediction received: {str(prediction)[:200]}")

        logger.info("✅ IC-default integration test PASSED")

    except Exception as e:
        logger.error(f"❌ IC-default integration test FAILED: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        _cleanup_resources(core_model, core_endpoint, endpoint_name, model_builder)


@pytest.mark.slow_test
def test_model_based_opt_out():
    """Integration test: Verify explicit MODEL_BASED opt-out works.

    1. Build JumpStart model
    2. Deploy WITH endpoint_type=MODEL_BASED
    3. Verify NO InferenceComponents on the endpoint
    4. Invoke endpoint
    5. Cleanup
    """
    core_model = None
    core_endpoint = None
    unique_id = str(uuid.uuid4())[:8]
    endpoint_name = f"{ENDPOINT_NAME_PREFIX}-mb-{unique_id}"

    try:
        # Build
        logger.info("Building JumpStart model for MODEL_BASED opt-out test...")
        compute = Compute(instance_type="ml.g5.2xlarge")
        jumpstart_config = JumpStartConfig(model_id=MODEL_ID)
        model_builder = ModelBuilder.from_jumpstart_config(
            jumpstart_config=jumpstart_config, compute=compute
        )

        core_model = model_builder.build(model_name=f"{MODEL_NAME_PREFIX}-mb-{unique_id}")
        logger.info(f"Model created: {core_model.model_name}")

        # Deploy WITH explicit MODEL_BASED
        logger.info("Deploying with endpoint_type=MODEL_BASED...")
        core_endpoint = model_builder.deploy(
            endpoint_name=endpoint_name,
            endpoint_type=EndpointType.MODEL_BASED,
        )
        logger.info(f"Endpoint created: {core_endpoint.endpoint_name}")

        # Verify NO InferenceComponents
        logger.info("Verifying no InferenceComponents (MODEL_BASED)...")
        sagemaker_client = model_builder.sagemaker_session.sagemaker_client
        ic_response = sagemaker_client.list_inference_components(
            EndpointNameEquals=endpoint_name
        )
        inference_components = ic_response.get("InferenceComponents", [])

        assert len(inference_components) == 0, (
            f"Expected MODEL_BASED endpoint with no InferenceComponents, but found "
            f"{len(inference_components)} on endpoint '{endpoint_name}'."
        )
        logger.info("✅ No InferenceComponents found (MODEL_BASED confirmed)")

        # Invoke endpoint
        logger.info("Invoking endpoint...")
        test_data = {"inputs": "What are falcons?", "parameters": {"max_new_tokens": 32}}
        result = core_endpoint.invoke(
            body=json.dumps(test_data), content_type="application/json"
        )
        prediction = json.loads(result.body.read().decode("utf-8"))
        logger.info(f"✅ Prediction received: {str(prediction)[:200]}")

        logger.info("✅ MODEL_BASED opt-out test PASSED")

    except Exception as e:
        logger.error(f"❌ MODEL_BASED opt-out test FAILED: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        _cleanup_resources(core_model, core_endpoint, endpoint_name, model_builder)


def _cleanup_resources(core_model, core_endpoint, endpoint_name, model_builder):
    """Clean up all AWS resources created during the test."""
    try:
        if core_endpoint:
            # Delete inference components first
            try:
                sagemaker_client = model_builder.sagemaker_session.sagemaker_client
                ic_response = sagemaker_client.list_inference_components(
                    EndpointNameEquals=endpoint_name
                )
                for ic in ic_response.get("InferenceComponents", []):
                    ic_name = ic["InferenceComponentName"]
                    logger.info(f"Deleting InferenceComponent: {ic_name}")
                    sagemaker_client.delete_inference_component(
                        InferenceComponentName=ic_name
                    )
            except Exception as e:
                logger.warning(f"Failed to delete inference components: {e}")

            # Delete endpoint
            try:
                core_endpoint.delete()
                logger.info(f"Deleted endpoint: {endpoint_name}")
            except Exception as e:
                logger.warning(f"Failed to delete endpoint: {e}")

            # Delete endpoint config
            try:
                endpoint_config = EndpointConfig.get(endpoint_config_name=endpoint_name)
                endpoint_config.delete()
                logger.info(f"Deleted endpoint config: {endpoint_name}")
            except Exception as e:
                logger.warning(f"Failed to delete endpoint config: {e}")

        if core_model:
            try:
                core_model.delete()
                logger.info(f"Deleted model: {core_model.model_name}")
            except Exception as e:
                logger.warning(f"Failed to delete model: {e}")

        logger.info("✅ Cleanup complete")
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
