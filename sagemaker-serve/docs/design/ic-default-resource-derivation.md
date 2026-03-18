# Design Doc: IC-Default Resource Derivation for ModelBuilder.deploy()

**Author:** Amit Modi  
**Date:** March 2026  
**Status:** Implementation  
**Version:** sagemaker-python-sdk 3.5.0 / sagemaker-serve 1.5.0  
**Branch:** `feature/ic-default-derive-resource-requirements`

---

## 1. Overview & Motivation

### Problem
Today, `ModelBuilder.deploy()` creates **MODEL_BASED** endpoints by default. Users who want Inference Component (IC)-based endpoints must:
1. Explicitly pass `endpoint_type=EndpointType.INFERENCE_COMPONENT_BASED`
2. Manually construct `ResourceRequirements(requests={"memory": ..., "num_cpus": ..., "num_accelerators": ..., "copies": ...})`

This requires users to know instance specs, GPU memory, and IC constraints — knowledge most users don't have.

### Solution
Make `INFERENCE_COMPONENT_BASED` the **default** endpoint type by auto-deriving `ResourceRequirements` using an 8-gate algorithm. The algorithm is defensive — it only enables IC when all compatibility gates pass, and falls back to MODEL_BASED gracefully on any failure.

### Benefits
- **Multi-model endpoints** become the default, enabling cost savings through resource sharing
- **Scale-to-zero** support for IC-based endpoints
- **No user action required** — existing `deploy()` calls automatically upgrade
- **Full backward compatibility** via opt-out mechanisms

---

## 2. Scope

| In Scope | Out of Scope |
|----------|--------------|
| PSDK v3.x (sagemaker-serve 1.5.0+) | PSDK v2.x (unaffected) |
| Standard JumpStart models | Custom containers with non-standard model servers |
| GPU instances (G5, G6, P3, P4, P5) | Trainium (ml.trn1) — future work |
| CPU instances (M6i, C6i, R6i) | Graviton (ml.g7g) — future work |
| Inferentia (Inf2) | |

---

## 3. The 8-Gate Algorithm

The algorithm is implemented in `sagemaker-serve/src/sagemaker/serve/utils/ic_utils.py`.

```
User calls ModelBuilder.deploy()
         │
    Gate 1: endpoint_type == MODEL_BASED? ──yes──► return None (MODEL_BASED)
         │ no / not specified
    Gate 1b: SAGEMAKER_IC_DEFAULT env/config == false? ──yes──► return None
         │ no
    Gate 2: Model IC-compatible? ──no──► raise ValueError
         │ yes                          (Pipeline models, Triton ensembles)
         │
    Gate 3: Instance family supported? ──no──► raise ValueError
         │ yes                               (ml.m5, ml.c5, ml.p2 excluded)
         │
    Gate 4: Region supported? ──no──► raise ValueError
         │ yes                       (GovCloud, China excluded)
         │
    Gate 5: Caller supplied ResourceRequirements? ──yes──► validate & return
         │ no
    Gate 6: initial_instance_count == 0? ──yes──► return scale-to-zero RR
         │ no
    Gate 7: JumpStart config fetch (5s timeout) ──success──► return JumpStart RR
         │ fail/timeout
    Gate 8: INSTANCE_SPEC_REGISTRY lookup ──found──► return derived RR
         │ not found
    raise ValueError("Cannot auto-derive ResourceRequirements")
```

### Gate Details

| Gate | Purpose | Failure Mode |
|------|---------|-------------|
| 1 | Explicit MODEL_BASED opt-out | Returns None (use MODEL_BASED) |
| 1b | Env/config opt-out | Returns None (use MODEL_BASED) |
| 2 | Model compatibility | ValueError (Pipeline, Triton) |
| 3 | Instance family support | ValueError (ml.m5, ml.c5, ml.p2) |
| 4 | Region support | ValueError (GovCloud, China) |
| 5 | Caller-supplied requirements | Validates & returns as-is |
| 6 | Scale-to-zero | Returns minimal RR with copies=0 |
| 7 | JumpStart metadata | Falls through on timeout/failure |
| 8 | Local instance registry | Returns derived RR or ValueError |

---

## 4. Integration Point

### Location
`ModelBuilder.deploy()` in `sagemaker-serve/src/sagemaker/serve/model_builder.py`

### Current Behavior
```python
# When no inference_config is provided:
deploy_kwargs = {
    "endpoint_type": EndpointType.MODEL_BASED,  # Hardcoded
    ...
}
```

### New Behavior
```python
# When no inference_config is provided:
derived_rr = self._try_derive_ic_resource_requirements(
    initial_instance_count=initial_instance_count
)
if derived_rr is not None:
    # IC-based deployment (new default)
    deploy_kwargs = {
        "endpoint_type": EndpointType.INFERENCE_COMPONENT_BASED,
        "resources": derived_rr,
        ...
    }
else:
    # MODEL_BASED fallback
    deploy_kwargs = {
        "endpoint_type": EndpointType.MODEL_BASED,
        ...
    }
```

### Error Handling
The `_try_derive_ic_resource_requirements()` helper wraps the 8-gate algorithm in a try/except:
- **`ValueError`** (expected from gates 2-4, 8): Caught, logged as INFO, returns None → MODEL_BASED
- **Other exceptions** (`ImportError`, `TypeError`, etc.): Propagated — indicates a bug that should not be silently swallowed

### User-Facing Log Message
When IC-default activates, users see:
```
INFO: Deploying with INFERENCE_COMPONENT_BASED endpoint (IC-default).
      ResourceRequirements: memory=24576 MB, cpus=4, accelerators=1, copies=1.
      To opt out, pass endpoint_type=EndpointType.MODEL_BASED or set SAGEMAKER_IC_DEFAULT=false.
```

---

## 5. Opt-Out Mechanisms

Users have three ways to opt out of IC-default:

### 5.1 Explicit Parameter
```python
model_builder.deploy(endpoint_type=EndpointType.MODEL_BASED)
```

### 5.2 Environment Variable
```bash
export SAGEMAKER_IC_DEFAULT=false
```

### 5.3 SageMaker Config YAML
```yaml
SageMaker:
  PythonSDK:
    InferenceComponentDefault: false
```

---

## 6. Instance Spec Registry

The `INSTANCE_SPEC_REGISTRY` in `ic_utils.py` provides a local lookup table for deriving ResourceRequirements when JumpStart metadata is unavailable. Coverage:

### GPU Instance Families
| Family | Sizes | GPU Memory (per GPU) |
|--------|-------|---------------------|
| ml.g5 | xlarge – 48xlarge | 24 GB |
| ml.g6 | xlarge – 48xlarge | 24 GB |
| ml.p3 | 2xlarge – 16xlarge | 16 GB |
| ml.p4d | 24xlarge | 40 GB |
| ml.p4de | 24xlarge | 80 GB |
| ml.p5 | 48xlarge | 80 GB |
| ml.inf2 | xlarge – 48xlarge | 32 GB |

### CPU Instance Families
| Family | Sizes | Memory Range |
|--------|-------|-------------|
| ml.m6i | large – 24xlarge | 8 – 384 GB |
| ml.c6i | large – 24xlarge | 4 – 192 GB |
| ml.r6i | large – 24xlarge | 16 – 768 GB |

### Unsupported Instance Families (raise ValueError at Gate 3)
- ml.m5, ml.m5d, ml.c5, ml.c5d, ml.p2

### Unsupported Regions (raise ValueError at Gate 4)
- us-gov-west-1, us-gov-east-1, us-iso-east-1, us-iso-west-1, us-isob-east-1
- cn-north-1, cn-northwest-1

---

## 7. ResourceRequirements API

The `ResourceRequirements` class (`sagemaker-core/src/sagemaker/core/resource_requirements.py`) is the data transfer object:

```python
rr = ResourceRequirements(
    requests={
        "num_cpus": 4,
        "memory": 24576,        # MB
        "num_accelerators": 1,
        "copies": 1,
    }
)

# Properties used by _deploy_core_endpoint():
rr.min_memory          # → 24576
rr.num_cpus            # → 4
rr.num_accelerators    # → 1
rr.copy_count          # → 1
rr.get_compute_resource_requirements()  # → dict for SageMaker API
```

**Compatibility verified:** ✅ The ic_utils module creates `ResourceRequirements` with the same constructor, and `_deploy_core_endpoint()` consumes it via `.get_compute_resource_requirements()` and `.copy_count`.

---

## 8. Testing Strategy

### 8.1 Unit Tests — Algorithm (Existing)

**File:** `sagemaker-serve/tests/unit/serve/utils/test_ic_utils.py`  
**Count:** 40+ tests  
**Coverage:**
- All 8 gates individually
- Gate priority/ordering
- Registry completeness validation
- Environment/config opt-out
- Edge cases (empty strings, None, malformed inputs)

**Run:**
```bash
python -m pytest sagemaker-serve/tests/unit/serve/utils/test_ic_utils.py -v
```

### 8.2 Unit Tests — Integration Point (New)

**File:** `sagemaker-serve/tests/unit/serve/test_model_builder_ic_default.py`  
**Coverage:**
- `_try_derive_ic_resource_requirements()` returns ResourceRequirements for supported configs
- `_try_derive_ic_resource_requirements()` returns None for unsupported configs
- `deploy()` creates IC-based endpoint when derivation succeeds
- `deploy()` creates MODEL_BASED endpoint when derivation fails
- Explicit `endpoint_type=MODEL_BASED` overrides IC-default
- ValueError from gates 2-4 triggers graceful fallback
- `model_server=None` edge case
- `endpoint_type=None` (new default) edge case
- Post-derivation validation (min_memory > 0)

**Run:**
```bash
python -m pytest sagemaker-serve/tests/unit/serve/test_model_builder_ic_default.py -v
```

### 8.3 Integration Tests — Real AWS (New)

**File:** `sagemaker-serve/tests/integ/test_ic_default_integration.py`  
**Infrastructure:** Real AWS (ml.g5.2xlarge, ~$0.25-0.50 per run)  
**Coverage:**

**Test 1: IC-default deployment**
1. Build JumpStart model (Falcon-7B)
2. Deploy WITHOUT `endpoint_type` → should auto-derive IC
3. Assert: `list_inference_components(EndpointNameEquals=...)` returns ≥ 1
4. Assert: InferenceComponent has correct resource requirements
5. Invoke endpoint with test prompt → assert valid prediction
6. Cleanup all resources

**Test 2: MODEL_BASED opt-out**
1. Build JumpStart model
2. Deploy WITH `endpoint_type=EndpointType.MODEL_BASED`
3. Assert: `list_inference_components(EndpointNameEquals=...)` returns 0
4. Invoke endpoint → assert valid prediction
5. Cleanup all resources

**Run:**
```bash
python -m pytest sagemaker-serve/tests/integ/test_ic_default_integration.py -v -s
```

### 8.4 CI/CD Considerations
- Unit tests: Run on every PR (fast, no AWS needed)
- Integration tests: Marked with `@pytest.mark.slow_test`, run in nightly pipeline
- All test resources tagged with `{"test": "ic-default", "cleanup": "safe-to-delete"}`

---

## 9. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Breaking change (MODEL_BASED → IC default) | High | 3 opt-out mechanisms; try/except fallback; INFO log message |
| Leaked AWS resources from failed tests | Medium | try/finally cleanup; resource tagging |
| Instance not in registry + JumpStart unreachable | Medium | Catches ValueError, falls back to MODEL_BASED |
| SIGALRM not available on Windows | Low | try/except in signal setup; falls through to gate 8 |
| Duplicated code in _deploy_core_endpoint | Low | Careful modification; comprehensive unit tests |

---

## 10. Future Work

- Add ml.trn1 (Trainium) and ml.g7g (Graviton) to INSTANCE_SPEC_REGISTRY
- Auto-scaling integration for IC-based endpoints
- Smart copy_count derivation based on instance_count and model size
- Telemetry: Track IC-default adoption rate vs opt-out rate
- Canary testing: Deploy IC-default to a subset of users first

---

## 11. Files Changed/Created

| File | Action | Description |
|------|--------|-------------|
| `docs/design/ic-default-resource-derivation.md` | New | This design doc |
| `src/sagemaker/serve/model_builder.py` | Modified | Wire ic_utils into deploy() |
| `src/sagemaker/serve/utils/__init__.py` | Modified | Export ic_utils |
| `tests/unit/serve/test_model_builder_ic_default.py` | New | Unit tests for integration |
| `tests/integ/test_ic_default_integration.py` | New | Real AWS integration tests |
