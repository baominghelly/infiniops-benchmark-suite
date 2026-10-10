# Low-Precision Policy and Recipe Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove known FP16/BF16-only reference failures from low-precision runs and repair the current recipe/schema failure block.

**Architecture:** Keep dtype policy in `full_operator_benchmark.py`, derive the PyTorch reference argument order from the resolved ATen schema by parameter name, and keep generated-wrapper argument order unchanged. Add narrowly scoped tensor recipes and input relations for index, pooling, convolution, loss, sparse, and pinv operators.

**Tech Stack:** Python `unittest`, the InfiniBench full-operator harness, and a MetaX FP32 targeted smoke run.

---

### Task 1: Expand FP32-only Policy

**Files:**
- Modify: `scripts/full_operator_benchmark.py`
- Modify: `tests/test_full_operator_benchmark.py`
- Modify: `docs/run_benchmark.md`

Add the 69 non-special operators whose September 21 failures were low-precision reference rejections to the existing FP32-only set. Verify that FP16/BF16 exclude those operators, while FP32 retains them.

### Task 2: Reorder Reference Arguments by ATen Schema

**Files:**
- Modify: `scripts/full_operator_benchmark.py`
- Modify: `tests/test_full_operator_benchmark.py`

Change `_resolve_aten_reference()` to return parameters in ATen schema order while retaining generated metadata types and keyword-only annotations. In `_run_schema_case()`, clone and pass values by parameter name rather than generated-wrapper position. Add a unit test where generated metadata places all tensors before scalars but ATen interleaves them.

### Task 3: Repair Index and Scatter Input Relations

**Files:**
- Modify: `scripts/full_operator_benchmark.py`
- Modify: `tests/test_full_operator_benchmark.py`

Derive `index`, `source`, and `src` tensors from `input` for `gather`, `index_add`, `index_copy`, `index_fill`, `index_reduce`, `index_select`, `scatter`, `scatter_add`, and `scatter_reduce`. Use a shared dimension and valid index bounds.

### Task 4: Repair Pool, Conv, and Activation Recipes

**Files:**
- Modify: `scripts/full_operator_benchmark.py`
- Modify: `tests/test_full_operator_benchmark.py`

Add explicit scalar overrides and random-sample recipes for fractional pools; ensure max-pool backward inputs are derived from their forward outputs; keep convolution kernel/stride lists and bias tensors compatible with generated metadata ordering.

### Task 5: Repair Loss and Normalization Recipes

**Files:**
- Modify: `scripts/full_operator_benchmark.py`
- Modify: `tests/test_full_operator_benchmark.py`

Set valid loss scalars and target/grad shapes for `multi_margin_loss`, `nll_loss`, and multilabel margin backward. Keep `batch_norm_elemt` channel-shaped parameters aligned.

### Task 6: Repair Sparse and Pinv Recipes

**Files:**
- Modify: `scripts/full_operator_benchmark.py`
- Modify: `tests/test_full_operator_benchmark.py`

Use compatible sparse COO inputs for `sspaddmm` and a positive-definite matrix for `linalg_pinv`.

### Task 7: Verify and Deploy

Run all local tests, run a no-GPU selection/manifest smoke, deploy the harness to all seven shared script hosts with backups, then rerun the 34-operator MetaX FP32 targeted set and compare statuses against `/tmp/recipe-schema-baseline`.

## Execution Result

- Completed on 2026-09-23.
- Local tests: 116/116 passed.
- Dtype policy on current MetaX metadata: float16 selects 373 and excludes 122; bfloat16 selects 373 and excludes 122; float32 selects 495 and excludes 0.
- MetaX targeted recipe run: 31/34 succeeded. `slow_conv3d`, `slow_conv3d_forward`, and `sspaddmm` remain backend runtime gaps rather than recipe/schema failures.
- Validation artifacts:
  `/data-aisoft/baoming/infinibench-recipe-validation/metax-c550/20260923/`.
