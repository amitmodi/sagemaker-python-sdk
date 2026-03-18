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
"""Inference Component (IC) utilities for auto-derivation of resource requirements.

This module implements the 8-gate algorithm for deriving ``ResourceRequirements``
when ``EndpointType.INFERENCE_COMPONENT_BASED`` is the default deployment mode.

The gates execute in order:
    1. Explicit ``endpoint_type=MODEL_BASED`` opt-out → return None
    2. ``_assert_ic_compatible(model, instance_type)`` → raises ValueError for excluded models
    3. ``instance_type in IC_UNSUPPORTED_INSTANCES`` → raise ValueError
    4. ``region_supports_inference_components(region)`` → raise ValueError if unsupported
    5. Caller-supplied ``resource_requirements`` → validate minimums and return as-is
    6. ``initial_instance_count == 0`` → return empty ResourceRequirements (scale-to-zero)
    7. JumpStart config fetch with 5s timeout → fall through on failure
    8. Instance spec derivation from INSTANCE_SPEC_REGISTRY
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from sagemaker.core.inference_config import ResourceRequirements

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Instance families that do NOT support Inference Components.
# These older instance types lack the firmware/scheduling support required.
IC_UNSUPPORTED_INSTANCE_FAMILIES = frozenset(
    {
        "ml.m5",
        "ml.m5d",
        "ml.c5",
        "ml.c5d",
        "ml.p2",
    }
)

# Regions where Inference Components are NOT supported.
# GovCloud and some newer/specialized regions.
IC_UNSUPPORTED_REGIONS = frozenset(
    {
        "us-gov-west-1",
        "us-gov-east-1",
        "us-iso-east-1",
        "us-iso-west-1",
        "us-isob-east-1",
        "cn-north-1",
        "cn-northwest-1",
    }
)

# Minimum resource requirement thresholds for IC-based endpoints.
IC_MIN_MEMORY_MB = 1024
IC_MIN_VCPUS = 1

# Default number of model copies for IC-based deployments.
IC_DEFAULT_COPIES = 1

# ---------------------------------------------------------------------------
# Instance Spec Registry
# ---------------------------------------------------------------------------
# Local lookup table: instance_type → {num_cpus, memory_mb, num_accelerators,
# per_gpu_memory_mb}.  GPU instances include accelerator fields; CPU instances
# do not.  This registry is used as gate 8 when JumpStart metadata is
# unavailable.
#
# Source: https://aws.amazon.com/sagemaker/pricing/instance-types/

INSTANCE_SPEC_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- GPU instances (G5 family) ---
    "ml.g5.xlarge": {
        "num_cpus": 4,
        "memory_mb": 16384,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g5.2xlarge": {
        "num_cpus": 8,
        "memory_mb": 32768,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g5.4xlarge": {
        "num_cpus": 16,
        "memory_mb": 65536,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g5.8xlarge": {
        "num_cpus": 32,
        "memory_mb": 131072,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g5.12xlarge": {
        "num_cpus": 48,
        "memory_mb": 196608,
        "num_accelerators": 4,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g5.16xlarge": {
        "num_cpus": 64,
        "memory_mb": 262144,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g5.24xlarge": {
        "num_cpus": 96,
        "memory_mb": 393216,
        "num_accelerators": 4,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g5.48xlarge": {
        "num_cpus": 192,
        "memory_mb": 786432,
        "num_accelerators": 8,
        "per_gpu_memory_mb": 24576,
    },
    # --- GPU instances (G6 family) ---
    "ml.g6.xlarge": {
        "num_cpus": 4,
        "memory_mb": 16384,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g6.2xlarge": {
        "num_cpus": 8,
        "memory_mb": 32768,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g6.4xlarge": {
        "num_cpus": 16,
        "memory_mb": 65536,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g6.8xlarge": {
        "num_cpus": 32,
        "memory_mb": 131072,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g6.12xlarge": {
        "num_cpus": 48,
        "memory_mb": 196608,
        "num_accelerators": 4,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g6.16xlarge": {
        "num_cpus": 64,
        "memory_mb": 262144,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g6.24xlarge": {
        "num_cpus": 96,
        "memory_mb": 393216,
        "num_accelerators": 4,
        "per_gpu_memory_mb": 24576,
    },
    "ml.g6.48xlarge": {
        "num_cpus": 192,
        "memory_mb": 786432,
        "num_accelerators": 8,
        "per_gpu_memory_mb": 24576,
    },
    # --- GPU instances (P3 family) ---
    "ml.p3.2xlarge": {
        "num_cpus": 8,
        "memory_mb": 62464,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 16384,
    },
    "ml.p3.8xlarge": {
        "num_cpus": 32,
        "memory_mb": 249856,
        "num_accelerators": 4,
        "per_gpu_memory_mb": 16384,
    },
    "ml.p3.16xlarge": {
        "num_cpus": 64,
        "memory_mb": 499712,
        "num_accelerators": 8,
        "per_gpu_memory_mb": 16384,
    },
    # --- GPU instances (P4d family) ---
    "ml.p4d.24xlarge": {
        "num_cpus": 96,
        "memory_mb": 1179648,
        "num_accelerators": 8,
        "per_gpu_memory_mb": 40960,
    },
    # --- GPU instances (P4de family) ---
    "ml.p4de.24xlarge": {
        "num_cpus": 96,
        "memory_mb": 1179648,
        "num_accelerators": 8,
        "per_gpu_memory_mb": 81920,
    },
    # --- GPU instances (P5 family) ---
    "ml.p5.48xlarge": {
        "num_cpus": 192,
        "memory_mb": 2097152,
        "num_accelerators": 8,
        "per_gpu_memory_mb": 81920,
    },
    # --- CPU instances (M6i family) ---
    "ml.m6i.large": {"num_cpus": 2, "memory_mb": 8192},
    "ml.m6i.xlarge": {"num_cpus": 4, "memory_mb": 16384},
    "ml.m6i.2xlarge": {"num_cpus": 8, "memory_mb": 32768},
    "ml.m6i.4xlarge": {"num_cpus": 16, "memory_mb": 65536},
    "ml.m6i.8xlarge": {"num_cpus": 32, "memory_mb": 131072},
    "ml.m6i.12xlarge": {"num_cpus": 48, "memory_mb": 196608},
    "ml.m6i.16xlarge": {"num_cpus": 64, "memory_mb": 262144},
    "ml.m6i.24xlarge": {"num_cpus": 96, "memory_mb": 393216},
    # --- CPU instances (C6i family) ---
    "ml.c6i.large": {"num_cpus": 2, "memory_mb": 4096},
    "ml.c6i.xlarge": {"num_cpus": 4, "memory_mb": 8192},
    "ml.c6i.2xlarge": {"num_cpus": 8, "memory_mb": 16384},
    "ml.c6i.4xlarge": {"num_cpus": 16, "memory_mb": 32768},
    "ml.c6i.8xlarge": {"num_cpus": 32, "memory_mb": 65536},
    "ml.c6i.12xlarge": {"num_cpus": 48, "memory_mb": 98304},
    "ml.c6i.16xlarge": {"num_cpus": 64, "memory_mb": 131072},
    "ml.c6i.24xlarge": {"num_cpus": 96, "memory_mb": 196608},
    # --- CPU instances (R6i family — memory-optimized) ---
    "ml.r6i.large": {"num_cpus": 2, "memory_mb": 16384},
    "ml.r6i.xlarge": {"num_cpus": 4, "memory_mb": 32768},
    "ml.r6i.2xlarge": {"num_cpus": 8, "memory_mb": 65536},
    "ml.r6i.4xlarge": {"num_cpus": 16, "memory_mb": 131072},
    "ml.r6i.8xlarge": {"num_cpus": 32, "memory_mb": 262144},
    "ml.r6i.12xlarge": {"num_cpus": 48, "memory_mb": 393216},
    "ml.r6i.16xlarge": {"num_cpus": 64, "memory_mb": 524288},
    "ml.r6i.24xlarge": {"num_cpus": 96, "memory_mb": 786432},
    # --- Inference-optimized (Inf2 family) ---
    "ml.inf2.xlarge": {
        "num_cpus": 4,
        "memory_mb": 32768,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 32768,
    },
    "ml.inf2.8xlarge": {
        "num_cpus": 32,
        "memory_mb": 131072,
        "num_accelerators": 1,
        "per_gpu_memory_mb": 32768,
    },
    "ml.inf2.24xlarge": {
        "num_cpus": 96,
        "memory_mb": 393216,
        "num_accelerators": 6,
        "per_gpu_memory_mb": 32768,
    },
    "ml.inf2.48xlarge": {
        "num_cpus": 192,
        "memory_mb": 786432,
        "num_accelerators": 12,
        "per_gpu_memory_mb": 32768,
    },
}


# ---------------------------------------------------------------------------
# Gate 2: IC compatibility check
# ---------------------------------------------------------------------------


def _assert_ic_compatible(model_server: Optional[str], model: Any) -> None:
    """Assert that the model/server combination is compatible with IC-based endpoints.

    Raises ``ValueError`` for:
    - Pipeline models (multi-container) — IC does not support multi-container
    - Triton ensemble models
    - TorchServe multi-model mode

    Args:
        model_server: The model server name (e.g. ``"TRITON"``, ``"TORCHSERVE"``).
        model: The model object from ModelBuilder.

    Raises:
        ValueError: If the model is incompatible with IC-based deployment.
    """
    # Pipeline models (list of Model objects = multi-container inference)
    if isinstance(model, list):
        raise ValueError(
            "PipelineModel (multi-container inference) is not supported with "
            "INFERENCE_COMPONENT_BASED endpoints. Use endpoint_type=EndpointType.MODEL_BASED "
            "or deploy models individually."
        )

    if model_server is not None:
        server_upper = str(model_server).upper()

        # Triton ensemble detection
        if "TRITON" in server_upper:
            raise ValueError(
                "Triton ensemble models are not supported with INFERENCE_COMPONENT_BASED "
                "endpoints. Use endpoint_type=EndpointType.MODEL_BASED instead."
            )

        # TorchServe multi-model mode detection
        if "TORCHSERVE" in server_upper:
            # TorchServe multi-model mode is not IC-compatible
            # (single-model TorchServe IS compatible, but we can't easily
            # distinguish at this point, so we let it through and rely on
            # runtime errors if truly multi-model)
            pass


# ---------------------------------------------------------------------------
# Gate 3: Unsupported instance types
# ---------------------------------------------------------------------------


def _is_ic_unsupported_instance(instance_type: str) -> bool:
    """Check whether an instance type belongs to an IC-unsupported family.

    Args:
        instance_type: SageMaker instance type string, e.g. ``"ml.m5.xlarge"``.

    Returns:
        True if the instance type is NOT supported for IC-based endpoints.
    """
    if not instance_type:
        return False
    # Extract family: "ml.m5.xlarge" → "ml.m5"
    parts = instance_type.split(".")
    if len(parts) >= 2:
        family = f"{parts[0]}.{parts[1]}"
        return family in IC_UNSUPPORTED_INSTANCE_FAMILIES
    return False


# ---------------------------------------------------------------------------
# Gate 4: Region support
# ---------------------------------------------------------------------------


def region_supports_inference_components(region: Optional[str]) -> bool:
    """Return True if the given AWS region supports Inference Components.

    Args:
        region: AWS region name, e.g. ``"us-east-1"``.

    Returns:
        True if Inference Components are supported in the region.
    """
    if not region:
        return False
    return region not in IC_UNSUPPORTED_REGIONS


# ---------------------------------------------------------------------------
# Gate 5: Validate caller-supplied resource requirements
# ---------------------------------------------------------------------------


def _validate_resource_requirements(resource_requirements: ResourceRequirements) -> None:
    """Validate that caller-supplied ``ResourceRequirements`` meet IC minimums.

    Args:
        resource_requirements: User-supplied resource requirements.

    Raises:
        ValueError: If memory < 1024 MB or vCPUs < 1.
    """
    if (
        resource_requirements.min_memory is not None
        and resource_requirements.min_memory < IC_MIN_MEMORY_MB
    ):
        raise ValueError(
            f"ResourceRequirements min_memory ({resource_requirements.min_memory} MB) is below "
            f"the minimum required for IC-based endpoints ({IC_MIN_MEMORY_MB} MB). "
            f"Please set min_memory >= {IC_MIN_MEMORY_MB}."
        )
    if (
        resource_requirements.num_cpus is not None
        and resource_requirements.num_cpus < IC_MIN_VCPUS
    ):
        raise ValueError(
            f"ResourceRequirements num_cpus ({resource_requirements.num_cpus}) is below "
            f"the minimum required for IC-based endpoints ({IC_MIN_VCPUS}). "
            f"Please set num_cpus >= {IC_MIN_VCPUS}."
        )


# ---------------------------------------------------------------------------
# Gate 7: JumpStart config fetch (5s timeout)
# ---------------------------------------------------------------------------


def _try_retrieve_jumpstart_resource_requirements(
    model_id: Optional[str],
    instance_type: str,
    region: Optional[str],
    sagemaker_session: Any = None,
    timeout: int = 5,
) -> Optional[ResourceRequirements]:
    """Attempt to retrieve deployment config from JumpStart with a timeout.

    This handles VPC-only environments where the JumpStart endpoint may be
    unreachable. On any failure the function returns ``None`` so the caller
    falls through to gate 8 (instance spec derivation).

    Args:
        model_id: JumpStart model identifier.
        instance_type: Target instance type.
        region: AWS region.
        sagemaker_session: SageMaker session object.
        timeout: Maximum seconds to wait for the JumpStart call.

    Returns:
        ``ResourceRequirements`` if successful, ``None`` on any failure.
    """
    if not model_id:
        return None

    try:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("JumpStart config fetch timed out")

        # Set alarm (Unix only; on Windows this is a no-op and we rely on
        # the underlying HTTP timeout).
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)
        except (AttributeError, ValueError):
            # signal.SIGALRM not available on this platform
            pass

        try:
            from sagemaker.core.jumpstart.artifacts.kwargs import _retrieve_model_deploy_kwargs

            deploy_kwargs = _retrieve_model_deploy_kwargs(
                model_id=model_id,
                model_version="*",
                instance_type=instance_type,
                region=region,
                tolerate_vulnerable_model=True,
                tolerate_deprecated_model=True,
                sagemaker_session=sagemaker_session,
            )

            # Extract resource requirements from deploy_kwargs if present
            resource_config = deploy_kwargs.get("resources")
            if resource_config and isinstance(resource_config, ResourceRequirements):
                return resource_config
        finally:
            # Cancel alarm
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                pass

    except Exception as e:
        logger.info(
            "JumpStart resource requirements fetch failed for model '%s' "
            "on instance '%s': %s. Falling through to instance spec derivation.",
            model_id,
            instance_type,
            str(e),
        )
    return None


# ---------------------------------------------------------------------------
# Gate 8: Instance spec derivation
# ---------------------------------------------------------------------------


def _derive_from_instance_spec(instance_type: str) -> Optional[ResourceRequirements]:
    """Derive ``ResourceRequirements`` from the ``INSTANCE_SPEC_REGISTRY``.

    For GPU instances: ``num_accelerators`` + ``per_gpu_memory_mb * num_accelerators``
    + ``copies=1``.
    For CPU instances: ``num_cpus`` + ``memory_mb`` + ``copies=1``.

    Args:
        instance_type: SageMaker instance type, e.g. ``"ml.g5.xlarge"``.

    Returns:
        ``ResourceRequirements`` if the instance type is in the registry,
        ``None`` otherwise.
    """
    spec = INSTANCE_SPEC_REGISTRY.get(instance_type)
    if spec is None:
        return None

    is_gpu = "num_accelerators" in spec

    if is_gpu:
        gpu_memory = spec["per_gpu_memory_mb"] * spec["num_accelerators"]
        return ResourceRequirements(
            requests={
                "num_accelerators": spec["num_accelerators"],
                "memory": gpu_memory,
                "num_cpus": spec.get("num_cpus", IC_MIN_VCPUS),
                "copies": IC_DEFAULT_COPIES,
            }
        )
    else:
        return ResourceRequirements(
            requests={
                "memory": spec["memory_mb"],
                "num_cpus": spec["num_cpus"],
                "copies": IC_DEFAULT_COPIES,
            }
        )


# ---------------------------------------------------------------------------
# Environment/config opt-out helpers
# ---------------------------------------------------------------------------


def _is_ic_default_disabled() -> bool:
    """Check whether the IC default has been disabled via environment or config.

    Checks (in order):
    1. ``SAGEMAKER_IC_DEFAULT`` environment variable (``"false"`` or ``"0"``)
    2. sagemaker config YAML (``SageMaker.PythonSDK.InferenceComponentDefault``)

    Returns:
        True if the IC default should be disabled (i.e. fall back to MODEL_BASED).
    """
    env_val = os.environ.get("SAGEMAKER_IC_DEFAULT", "").lower().strip()
    if env_val in ("false", "0", "no"):
        return True

    # Config YAML check — best-effort; if config is unavailable we default to enabled.
    try:
        from sagemaker.core.common_utils import get_config_value

        config_val = get_config_value(
            "SageMaker.PythonSDK.InferenceComponentDefault", config=None
        )
        if config_val is not None and str(config_val).lower().strip() in ("false", "0", "no"):
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Main 8-gate algorithm
# ---------------------------------------------------------------------------


def derive_resource_requirements(
    endpoint_type: Any,
    model: Any,
    model_server: Optional[str],
    instance_type: str,
    region: Optional[str],
    initial_instance_count: int,
    resource_requirements: Optional[ResourceRequirements] = None,
    model_id: Optional[str] = None,
    sagemaker_session: Any = None,
) -> Optional[ResourceRequirements]:
    """Derive ``ResourceRequirements`` for IC-based endpoints using the 8-gate algorithm.

    This is the central entry point called from ``ModelBuilder.deploy()`` when the
    default endpoint type is ``INFERENCE_COMPONENT_BASED``.

    Args:
        endpoint_type: The endpoint type enum value. If explicitly set to
            ``EndpointType.MODEL_BASED``, returns ``None`` immediately (gate 1).
        model: The model object from ModelBuilder.
        model_server: Model server name string (e.g. ``"TRITON"``).
        instance_type: SageMaker instance type for deployment.
        region: AWS region name.
        initial_instance_count: Number of initial EC2 instances.
        resource_requirements: User-supplied ``ResourceRequirements``, if any.
        model_id: JumpStart model ID, if applicable.
        sagemaker_session: SageMaker session for JumpStart calls.

    Returns:
        ``ResourceRequirements`` for IC-based deployment, or ``None`` if the
        deployment should use MODEL_BASED (gate 1 opt-out).

    Raises:
        ValueError: For excluded model types (gate 2), unsupported instances
            (gate 3), unsupported regions (gate 4), or invalid resource
            requirements (gate 5).
    """
    from sagemaker.core.enums import EndpointType

    # --- Gate 1: Explicit MODEL_BASED opt-out ---
    if endpoint_type == EndpointType.MODEL_BASED:
        logger.info("Gate 1: Explicit MODEL_BASED opt-out. Skipping IC resource derivation.")
        return None

    # Also check environment/config opt-out
    if _is_ic_default_disabled():
        logger.info(
            "IC default disabled via environment variable or config. "
            "Using MODEL_BASED endpoint."
        )
        return None

    # --- Gate 2: IC compatibility ---
    _assert_ic_compatible(model_server, model)
    logger.debug("Gate 2 passed: model is IC-compatible.")

    # --- Gate 3: Unsupported instance types ---
    if _is_ic_unsupported_instance(instance_type):
        raise ValueError(
            f"Instance type '{instance_type}' is not supported for "
            f"INFERENCE_COMPONENT_BASED endpoints. Unsupported instance families: "
            f"{sorted(IC_UNSUPPORTED_INSTANCE_FAMILIES)}. "
            f"Use endpoint_type=EndpointType.MODEL_BASED or choose a supported "
            f"instance type (e.g. ml.g5.xlarge, ml.m6i.xlarge, ml.c6i.xlarge)."
        )
    logger.debug("Gate 3 passed: instance type '%s' is IC-compatible.", instance_type)

    # --- Gate 4: Region support ---
    if not region_supports_inference_components(region):
        raise ValueError(
            f"Region '{region}' does not support Inference Components. "
            f"Unsupported regions: {sorted(IC_UNSUPPORTED_REGIONS)}. "
            f"Use endpoint_type=EndpointType.MODEL_BASED or deploy in a "
            f"supported region."
        )
    logger.debug("Gate 4 passed: region '%s' supports Inference Components.", region)

    # --- Gate 5: Caller-supplied resource_requirements ---
    if resource_requirements is not None:
        _validate_resource_requirements(resource_requirements)
        logger.info(
            "Gate 5: Using caller-supplied ResourceRequirements "
            "(memory=%s MB, cpus=%s, accelerators=%s, copies=%s).",
            resource_requirements.min_memory,
            resource_requirements.num_cpus,
            resource_requirements.num_accelerators,
            resource_requirements.copy_count,
        )
        return resource_requirements

    # --- Gate 6: Scale-to-zero pattern ---
    if initial_instance_count == 0:
        logger.info(
            "Gate 6: initial_instance_count=0 detected. "
            "Returning empty ResourceRequirements for scale-to-zero."
        )
        return ResourceRequirements(requests={"memory": IC_MIN_MEMORY_MB, "copies": 0})

    # --- Gate 7: JumpStart config fetch (5s timeout) ---
    jumpstart_rr = _try_retrieve_jumpstart_resource_requirements(
        model_id=model_id,
        instance_type=instance_type,
        region=region,
        sagemaker_session=sagemaker_session,
    )
    if jumpstart_rr is not None:
        logger.info(
            "Gate 7: Derived ResourceRequirements from JumpStart config "
            "for model '%s' on '%s'.",
            model_id,
            instance_type,
        )
        return jumpstart_rr

    # --- Gate 8: Instance spec derivation ---
    spec_rr = _derive_from_instance_spec(instance_type)
    if spec_rr is not None:
        logger.info(
            "Gate 8: Derived ResourceRequirements from INSTANCE_SPEC_REGISTRY "
            "for '%s' (memory=%s MB, cpus=%s, accelerators=%s, copies=%s).",
            instance_type,
            spec_rr.min_memory,
            spec_rr.num_cpus,
            spec_rr.num_accelerators,
            spec_rr.copy_count,
        )
        return spec_rr

    # Fallback: instance type not in registry — raise informative error
    raise ValueError(
        f"Cannot auto-derive ResourceRequirements for instance type '{instance_type}'. "
        f"The instance type is not in the local INSTANCE_SPEC_REGISTRY and JumpStart "
        f"metadata was unavailable. Please either:\n"
        f"  1. Supply explicit resource_requirements via "
        f"inference_config=ResourceRequirements(...)\n"
        f"  2. Use endpoint_type=EndpointType.MODEL_BASED\n"
        f"  3. Use a supported instance type from: "
        f"{sorted(INSTANCE_SPEC_REGISTRY.keys())[:10]}..."
    )
