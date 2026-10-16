# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Numerical and autograd contracts for fused shortcut A/B boundaries."""

import pytest
import torch

from megatron.core.fusions.wide_residual_shortcut import (
    boundary_norm,
    validate_input,
    write_read_norm,
)


def inputs(streams, readers, hidden=96, tokens=1024):
    torch.manual_seed(917)
    residual = torch.randn(tokens, streams * hidden, device="cuda", requires_grad=True)
    updates = tuple(
        torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(readers)
    )
    controllers = tuple(
        torch.randn(streams + 2, device="cuda", requires_grad=True) for _ in range(2 + readers)
    )
    gammas = tuple(
        torch.randn(hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(readers)
    )
    return residual, updates, controllers, gammas


def run(raw, streams):
    residual, updates, controllers, gammas = raw
    if len(updates) == 1:
        return write_read_norm(residual, updates[0], *controllers, gammas[0], num_streams=streams)
    return boundary_norm(
        residual, *updates, *controllers[:2], controllers[2:], gammas, num_streams=streams
    )


def reference(raw, streams):
    """Independent FP64 formula with the specified BF16 branch-sum/output boundaries."""
    residual, updates, controllers, gammas = raw
    hidden = updates[0].shape[-1]
    r = residual.double().view(-1, streams, hidden)
    w = 2 * controllers[0][:streams].double().sigmoid()
    t = (
        torch.ones_like(w)
        if controllers[1] is None
        else 1 - 0.1 * (-controllers[1][:streams].double()).sigmoid()
    )
    u = updates[0] if len(updates) == 1 else (updates[0].float() + updates[1].float()).bfloat16()
    current = t[None, :, None] * r + w[None, :, None] * u.double()[:, None, :]
    outputs = []
    for read, gamma in zip(controllers[2:], gammas):
        x = (read[:streams].double().sigmoid()[None, :, None] * current).sum(1)
        rs = (x.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
        outputs.append(((x * rs) * gamma.double()).bfloat16())
    return current.flatten(1).float(), *outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fusion")
@pytest.mark.parametrize("streams", range(1, 7))
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("hidden,tokens", [(32, 1024), (96, 1024), (128, 2048)])
def test_forward_and_complete_backward(streams, readers, hidden, tokens):
    raw = inputs(streams, readers, hidden, tokens)
    originals = (raw[0], *raw[1], *raw[2], *raw[3])
    cloned = tuple(t.detach().clone().requires_grad_(True) for t in originals)
    n = readers
    ref_raw = (cloned[0], cloned[1 : 1 + n], cloned[1 + n : 3 + 2 * n], cloned[3 + 2 * n :])
    actual, expected = run(raw, streams), reference(ref_raw, streams)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-6, atol=2e-6)
    for a, b in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(a, b, rtol=1 / 128, atol=1 / 128)
    dys = tuple(torch.randn_like(t) for t in actual)
    ga = torch.autograd.grad(actual, originals, dys)
    ge = torch.autograd.grad(expected, cloned, dys)
    for index, (a, b) in enumerate(zip(ga, ge)):
        if index == 0:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
        elif originals[index].dtype == torch.bfloat16:
            torch.testing.assert_close(a, b, rtol=1 / 128, atol=1 / 128)
        else:
            torch.testing.assert_close(a, b, rtol=5e-4, atol=2e-4)
    for c in raw[2]:
        index = next(i for i, p in enumerate(originals) if p is c)
        assert torch.count_nonzero(ga[index][streams:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fusion")
@pytest.mark.parametrize("streams", [2, 3, 6])
@pytest.mark.parametrize("route", [0, 1, 2])
@pytest.mark.parametrize("retention", [False, True])
def test_unused_outputs_and_frozen_parameters(streams, route, retention):
    raw = inputs(streams, 2)
    raw[2][1].requires_grad_(False)
    raw[3][1].requires_grad_(False)
    if not retention:
        raw = (raw[0], raw[1], (raw[2][0], None, *raw[2][2:]), raw[3])
    result = run(raw, streams)
    result[route].float().sum().backward()
    assert raw[0].grad is not None
    assert (raw[2][1] is None or raw[2][1].grad is None) and raw[3][1].grad is None
    if not retention and route == 0:
        assert torch.equal(raw[0].grad, torch.ones_like(raw[0]))
    for t in (raw[0], *raw[1], *raw[2], *raw[3]):
        if t is not None and t.grad is not None:
            assert torch.isfinite(t.grad).all()


def test_reject_bf16_residual_setting():
    with pytest.raises(ValueError, match="fp32_residual_connection=True"):
        validate_input(
            torch.empty(1024, 96, dtype=torch.bfloat16), 32, 3, fp32_residual_connection=False
        )


@pytest.mark.parametrize(
    "shape,hidden,streams,match",
    [
        ((1023, 96), 32, 3, "T"),
        ((1025, 96), 32, 3, "T"),
        ((1024, 99), 33, 3, "H"),
        ((1024, 96), 32, 0, "stream"),
        ((1024, 96), 32, 7, "stream"),
        ((1024, 96), 32, True, "stream"),
        ((1024, 95), 32, 3, "S \\* H"),
        ((17408, 9216), 3072, 3, "CUDA"),
        ((17408, 30720), 10240, 3, "at most"),
    ],
)
def test_invalid_shapes_fail_before_cuda(shape, hidden, streams, match):
    # Meta tensors validate metadata without allocating production-size CPU tensors.
    with pytest.raises(ValueError, match=match):
        validate_input(
            torch.empty(shape, device="meta"), hidden, streams, fp32_residual_connection=True
        )


def test_shared_device_source_manifest():
    import ast
    import hashlib
    import json
    from pathlib import Path

    from megatron.core.fusions import wide_residual_shortcut

    root = Path(wide_residual_shortcut.__file__).parent
    baseline = json.loads((root / "provenance.json").read_text())["shared_device_functions"]

    def canonical(node):
        if isinstance(node, ast.AST):
            return [
                type(node).__name__,
                [(k, canonical(v)) for k, v in ast.iter_fields(node) if k != "type_params"],
            ]
        if isinstance(node, list):
            return [canonical(v) for v in node]
        return node

    for name, expected in baseline.items():
        tree = ast.parse((root / name).read_text())
        actual = {
            n.name: hashlib.sha256(json.dumps(canonical(n), sort_keys=True).encode()).hexdigest()
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in expected
        }
        assert actual == expected, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fusion")
@pytest.mark.parametrize("streams", range(1, 7))
@pytest.mark.parametrize("readers", [1, 2])
def test_disabled_retention_gradients_and_replay(streams, readers):
    raw = inputs(streams, readers, 128, 1024)
    residual, updates, controllers, gammas = raw
    controllers = (controllers[0], None, *controllers[2:])
    raw = residual, updates, controllers, gammas
    originals = (residual, *updates, *(c for c in controllers if c is not None), *gammas)
    clone = lambda t: t.detach().clone().requires_grad_(True)
    ref_raw = (
        clone(residual),
        tuple(map(clone, updates)),
        tuple(None if c is None else clone(c) for c in controllers),
        tuple(map(clone, gammas)),
    )
    ref_inputs = (ref_raw[0], *ref_raw[1], *(c for c in ref_raw[2] if c is not None), *ref_raw[3])
    actual, expected = run(raw, streams), reference(ref_raw, streams)
    dys = tuple(torch.randn_like(t) for t in actual)
    ga = torch.autograd.grad(actual, originals, dys)
    ge = torch.autograd.grad(expected, ref_inputs, dys)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-6, atol=2e-6)
    for a, b in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(a, b, rtol=1 / 128, atol=1 / 128)
    for original, a, b in zip(originals, ga, ge):
        if original.dtype == torch.bfloat16:
            torch.testing.assert_close(a, b, rtol=1 / 128, atol=1 / 128)
        else:
            torch.testing.assert_close(a, b, rtol=5e-4, atol=2e-4)
    repeated = run(raw, streams)
    repeated_grads = torch.autograd.grad(repeated, originals, dys)
    assert all(torch.equal(a, b) for a, b in zip(actual, repeated))
    assert all(torch.equal(a, b) for a, b in zip(ga, repeated_grads))


@pytest.mark.parametrize(
    "name",
    [
        "reconstruction",
        "reconstruction_streams",
        "reconstruction_backward_streams",
        "reconstruction_cooperative_native",
        "reconstruction_cooperative_streams",
        "reconstruction_joint_gluon",
        "reconstruction_prepare_gluon",
        "reconstruction_stream_reductions",
        "reconstruction_dispatch",
        "fusion_runtime",
    ],
)
def test_standalone_uses_canonical_module(name, monkeypatch):
    import importlib
    from pathlib import Path

    standalone = Path(__file__).resolve().parents[4] / "shortcut_block_bench"
    if not standalone.is_dir():
        pytest.skip("standalone benchmark checkout is not present")
    monkeypatch.syspath_prepend(str(standalone))
    packaged = importlib.import_module("megatron.core.fusions.wide_residual_shortcut." + name)
    assert importlib.import_module(name) is packaged


@pytest.mark.parametrize("streams", [2, 3, 6])
@pytest.mark.parametrize("retention_logit", [None, -80.0, 0.0, 4.59511985, 80.0])
@pytest.mark.parametrize("hidden", [64, 96, 128])
def test_mixed_inverse_chain_bounded_retention(streams, retention_logit, hidden):
    """Worst permitted retention and identity carry stay stable over 16 writes."""
    from contextlib import nullcontext

    from megatron.core.fusions.wide_residual_shortcut.residual_inverse import ResidualInverseManager
    from megatron.core.transformer.streamwise_residual_ops import streamwise_sigmoid_writeback

    torch.manual_seed(55)
    r = torch.randn(1024, streams * hidden, device="cuda", requires_grad=True)
    updates = tuple(
        torch.randn(1024, hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(16)
    )
    w = torch.zeros(128, device="cuda", requires_grad=True)
    t = (
        None
        if retention_logit is None
        else torch.full((128,), retention_logit, device="cuda", requires_grad=True)
    )
    read = torch.zeros(128, device="cuda", requires_grad=True)
    gamma = torch.ones(hidden, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def execute(inverse):
        original = (r, *updates, w, *((t,) if t is not None else ()), read, gamma)
        values = [v.detach().clone().requires_grad_(True) for v in original]
        x, us, ww = values[0], values[1:17], values[17]
        tt = None if t is None else values[18]
        rr, gg = values[-2:]
        with ResidualInverseManager(16).activate() if inverse else nullcontext():
            for i, u in enumerate(us):
                if i % 2:
                    x = streamwise_sigmoid_writeback(
                        x, u, ww, streams, retention_logits=tt, retention_max_forget=0.1
                    )
                else:
                    x, _ = write_read_norm(x, u, ww, tt, rr, gg, num_streams=streams)
        x.square().mean().backward()
        return x.detach(), [v.grad for v in values]

    actual, grads = execute(True)
    expected, reference_grads = execute(False)
    assert torch.equal(actual, expected)
    for a, b in zip(grads, reference_grads):
        if a is None or b is None:
            assert a is None and b is None
        else:
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
