# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for the activation-strategy runtime on a small owner module."""

import copy

import pytest
import torch
from torch import nn

from megatron.core.activation_policy import (
    CheckpointStrategy,
    CheckpointWithoutOutputStrategy,
    Event,
    OffloadStrategy,
    ReplayRegion,
    ResidualBranch,
    TensorRef,
    has_strategy,
    install_activation_strategies,
    installed_strategies,
    mark_borrowed,
)
from megatron.core.activation_policy import runtime as runtime_module
from megatron.core.activation_policy import strategies as strategies_module
from megatron.core.activation_policy.recompute import replay as replay_module
from megatron.core.tensor_parallel.random import initialize_rng_tracker
from megatron.core.transformer.module import _ModulePointHook
from tests.unit_tests.test_utilities import Utils, install_residual_replay


class _Owner(nn.Module):
    """norm -> linear branch -> residual add, the shape of a transformer sub-block."""

    def __init__(self, hidden=16):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, hidden)
        self.training = True

    def _bda(self, branch_out, residual):
        return branch_out + residual

    def forward(self, x):
        normed = self.norm(x)
        branch = self.linear(normed)
        return self._bda(branch, x)

    def get_residual_branches_by_layer(self):
        """For residual replay: the norm (an op) and ``_bda`` (the write) of one layer."""
        return [
            [
                ResidualBranch(
                    ops=(ReplayRegion(self, self.norm),), write=ReplayRegion(self, self._bda)
                )
            ]
        ]


def _run(owner, x):
    x = x.detach().clone().requires_grad_(True)
    out = owner(x)
    out.sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in owner.named_parameters()}
    return out.detach(), x.grad.detach(), grads


@pytest.fixture
def parallel_state():
    Utils.initialize_model_parallel()
    initialize_rng_tracker(force_reset=True)
    yield
    Utils.destroy_model_parallel()


def _state(owner):
    return runtime_module._installed(owner).state


def _install(owner, *strategies):
    install_activation_strategies(owner, strategies)


def _recompute_norm(owner):
    return CheckpointWithoutOutputStrategy(owner.norm, Event(owner._bda, "enter"))


def _replay(*owners, block_size=1):
    """Install residual replay of the owners (one layer step each) on a test stack; returns the
    stack (run owner ``i`` as ``stack._run_layer(i, x)``) and the strategy."""
    stack = install_residual_replay(owners, block_size=block_size)
    return stack, stack.replay


class TestInstall:
    def test_without_strategies_nothing_is_hooked(self):
        owner = _Owner()
        x = torch.randn(2, 16)
        torch.testing.assert_close(owner(x), owner.linear(owner.norm(x)) + x)
        assert installed_strategies(owner) == {}
        assert "_bda" not in owner.__dict__ and "forward" not in owner.norm.__dict__

    def test_method_points_are_hooked_transparently(self):
        owner = _Owner()
        bound = owner._bda
        _install(owner, CheckpointStrategy(owner._bda, impl="tensor_parallel"))
        hook = owner.__dict__["_bda"]
        # The hook stands for the bound method: the same point before and after install.
        assert hook == bound and hash(hook) == hash(bound)
        assert has_strategy(owner, owner._bda, CheckpointStrategy)
        assert has_strategy(owner, bound, CheckpointStrategy)
        assert owner._bda(torch.ones(2), torch.ones(2)).tolist() == [2.0, 2.0]
        runtime_module._uninstall(owner)
        assert "_bda" not in owner.__dict__

    def test_install_wraps_named_children_and_uninstall_restores(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner))
        assert "forward" in owner.norm.__dict__
        assert "forward" not in owner.linear.__dict__  # not observed
        runtime_module._uninstall(owner)
        assert "forward" not in owner.norm.__dict__
        assert installed_strategies(owner) == {}

    def test_points_given_by_name_are_rejected(self):
        owner = _Owner()
        with pytest.raises(TypeError, match="child module or a method"):
            _install(owner, CheckpointStrategy("_bda", impl="tensor_parallel"))

    def test_deepcopy_remaps_owner(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner))
        clone = copy.deepcopy(owner)
        wrapper = clone.norm.__dict__["forward"]
        assert wrapper.owner is clone and wrapper.module is clone.norm

    def test_deepcopy_of_child_alone_gets_plain_forward(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner))
        norm = copy.deepcopy(owner.norm)
        assert not isinstance(norm.__dict__["forward"], _ModulePointHook)
        x = torch.randn(2, 16)
        assert torch.equal(norm(x), owner.norm(x))

    def test_uninstall_restores_instance_forward(self):
        owner = _Owner()
        patched = owner.norm.forward
        owner.norm.forward = patched
        _install(owner, _recompute_norm(owner))
        runtime_module._uninstall(owner)
        assert owner.norm.__dict__["forward"] is patched

    def test_conflicting_strategies_raise(self):
        owner = _Owner()
        with pytest.raises(ValueError):
            _install(owner, _recompute_norm(owner), CheckpointStrategy(owner.norm))

    def test_install_replaces_previous_strategies(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner))
        install_activation_strategies(owner, [CheckpointStrategy(owner.linear)])
        assert "forward" not in owner.norm.__dict__
        assert "forward" in owner.linear.__dict__
        assert list(installed_strategies(owner)) == [owner.linear]

    def test_replay_orders_before_recompute(self):
        owner = _Owner()
        _install(owner, _recompute_norm(owner))
        _replay(owner)
        assert [type(s) for s in installed_strategies(owner)[owner.norm]] == [
            replay_module._ReplayHook,
            CheckpointWithoutOutputStrategy,
        ]

    def test_contributed_hooks_survive_owner_reinstall_and_are_withdrawn(self):
        owner = _Owner()
        stack, _ = _replay(owner)
        _install(owner, _recompute_norm(owner))
        assert replay_module._ReplayHook in [
            type(s) for s in installed_strategies(owner)[owner.norm]
        ]
        # The layer output of the block's last layer stays live: it has no replay hook.
        assert owner._bda not in installed_strategies(owner)
        install_activation_strategies(stack, [])
        assert [type(s) for s in installed_strategies(owner)[owner.norm]] == [
            CheckpointWithoutOutputStrategy
        ]
        assert owner._bda not in installed_strategies(owner)


@pytest.mark.usefixtures("parallel_state")
class TestPolicies:
    def _reference(self, x):
        torch.manual_seed(0)
        owner = _Owner().cuda()
        return owner, _run(owner, x)

    def test_recompute_releases_output_and_matches(self):
        x = torch.randn(8, 4, 16, device="cuda")
        owner, (ref_out, ref_dx, ref_grads) = self._reference(x)

        _install(owner, _recompute_norm(owner))
        for p in owner.parameters():
            p.grad = None

        seen = {}
        original = owner.norm.__dict__["forward"].forward

        def spy(*args, **kwargs):
            out = original(*args, **kwargs)
            seen.setdefault("out", out)
            return out

        owner.norm.__dict__["forward"].forward = spy
        xi = x.detach().clone().requires_grad_(True)
        out = owner(xi)
        # The norm output was released when the branch output reached the residual add.
        assert seen["out"].untyped_storage().size() == 0
        out.sum().backward()

        assert torch.equal(out.detach(), ref_out)
        assert torch.equal(xi.grad, ref_dx)
        for name, grad in ref_grads.items():
            assert torch.equal(dict(owner.named_parameters())[name].grad, grad), name
        assert not _state(owner).pending

    def test_residual_replay_releases_block_and_matches(self):
        x = torch.randn(8, 4, 16, device="cuda")
        torch.manual_seed(0)
        ref = nn.Sequential(_Owner(), _Owner()).cuda()
        ref_out, ref_dx, ref_grads = _run(ref, x)

        model = copy.deepcopy(ref)
        for p in model.parameters():
            p.grad = None
        stack, strategy = _replay(*model, block_size=2)
        replayed = replay_module._ReplayHook
        assert has_strategy(model[0], model[0].norm, replayed)
        assert has_strategy(model[0], model[0]._bda, replayed)
        # The block's last write is its live layer output.
        assert has_strategy(model[1], model[1].norm, replayed)
        assert not has_strategy(model[1], model[1]._bda, replayed)

        seen = []
        original = model[0].norm.__dict__["forward"].forward

        def spy(*args, **kwargs):
            out = original(*args, **kwargs)
            seen.append(out)
            return out

        model[0].norm.__dict__["forward"].forward = spy
        xi = x.detach().clone().requires_grad_(True)
        out = stack._run_layer(1, stack._run_layer(0, xi))
        # The block's last step (owner 1) released the block's replayed outputs.
        assert seen[0].untyped_storage().size() == 0
        assert strategy.managers == [None]
        out.sum().backward()
        assert torch.equal(out.detach(), ref_out)
        assert torch.equal(xi.grad, ref_dx)
        for name, grad in ref_grads.items():
            assert torch.equal(dict(model.named_parameters())[name].grad, grad), name

        model.eval()
        with torch.no_grad():
            stack._run_layer(1, stack._run_layer(0, xi))
        assert strategy.managers == [None]  # nothing replayed outside training

    def test_checkpoint_with_tensor_kwargs_matches(self):
        class KwOwner(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)

            def _branch(self, x, *, scale, mode):
                assert mode == "mul"
                return self.linear(x) * scale

            def forward(self, x, scale):
                return self._branch(x, scale=scale, mode="mul")

        torch.manual_seed(0)
        owner = KwOwner().cuda()
        x = torch.randn(4, 16, device="cuda", requires_grad=True)
        scale = torch.randn(4, 16, device="cuda", requires_grad=True)
        owner(x, scale).sum().backward()
        ref = (x.grad.clone(), scale.grad.clone(), owner.linear.weight.grad.clone())
        x.grad = scale.grad = owner.linear.weight.grad = owner.linear.bias.grad = None

        install_activation_strategies(
            owner, [CheckpointStrategy(owner._branch, impl="tensor_parallel")]
        )
        owner(x, scale).sum().backward()
        assert torch.equal(x.grad, ref[0])
        assert torch.equal(scale.grad, ref[1])
        assert torch.equal(owner.linear.weight.grad, ref[2])


class _FakeOffloadInterface:
    """Stand-in for FineGrainedActivationOffloadingInterface that logs calls."""

    log = []

    def __init__(self, offload, tensor, name):
        self.tensor, self.name = tensor, name

    def __enter__(self):
        self.log.append(("start", self.name))
        return self.tensor

    def __exit__(self, *args):
        self.log.append(("close", self.name))

    def group_offload(self, tensor, forced_released_tensors=None, delay_offload=False):
        self.log.append(("commit", self.name, tuple(id(t) for t in forced_released_tensors)))
        return tensor

    @staticmethod
    def mark_not_offload(tensor):
        _FakeOffloadInterface.log.append(("not_offload",))


@pytest.fixture
def fake_offload(monkeypatch):
    _FakeOffloadInterface.log = []
    monkeypatch.setattr(strategies_module, "_offload_interface", lambda: _FakeOffloadInterface)
    return _FakeOffloadInterface


class TestOffloadEvents:
    def test_group_spans_regions_and_commits_lifo(self, fake_offload):
        """The fused_group_mlp shape: outer group over norm+linear, inner group on linear."""
        owner = _Owner()
        _install(
            owner,
            OffloadStrategy(
                owner.norm,
                "outer",
                commit_on=owner._bda,
                context_until=owner.linear,
                release=(TensorRef(owner.norm, "input", 0), TensorRef(owner._bda, "input", 1)),
            ),
            OffloadStrategy(owner.linear, "inner", commit_on=owner._bda),
        )
        x = torch.randn(2, 16)
        owner(x)
        log = fake_offload.log
        assert [e[:2] for e in log] == [
            ("start", "outer"),
            ("start", "inner"),
            ("close", "inner"),
            ("close", "outer"),
            ("commit", "inner"),
            ("commit", "outer"),
        ]
        # Both references resolve to the layer input (norm input and bda residual).
        assert log[-1][2] == (id(x), id(x))
        state = _state(owner)
        assert not state.records and not state.pending and state.idle

    def test_skip_when_replayed_and_training_only(self, fake_offload):
        owner = _Owner()
        _install(
            owner,
            OffloadStrategy(
                owner.norm, "g", commit_on=owner._bda, training_only=True, skip_when_replayed=True
            ),
        )
        _replay(owner)
        x = torch.randn(2, 16, requires_grad=True)
        owner(x)  # the norm is replayed: not offloaded
        owner.eval()
        owner(x)  # not training: not offloaded
        assert fake_offload.log == []

    def test_fired_enter_keeps_exit_actions(self, fake_offload):
        owner = _Owner()
        release = (TensorRef.input(owner._bda, 1),)
        _install(owner, OffloadStrategy(owner.norm, "g", commit_on=owner._bda, release=release))
        x = torch.randn(2, 16)
        owner.norm(x)  # an executor runs the norm directly
        runtime_module.fire_boundary(owner, owner._bda, "enter", [])
        assert [e[0] for e in fake_offload.log] == ["start", "close"]
        y = runtime_module.fire_boundary(owner, owner._bda, "exit", x, inputs=(x, x))
        assert y is x
        assert fake_offload.log[-1] == ("commit", "g", (id(x),))
        assert not _state(owner).records and not _state(owner).armed

    def test_records_only_for_open_groups(self, fake_offload):
        owner = _Owner()
        release = (TensorRef.input(owner._bda, 1),)
        _install(
            owner,
            OffloadStrategy(
                owner.norm, "g", commit_on=owner._bda, release=release, training_only=True
            ),
        )
        owner.eval()
        owner(torch.randn(2, 16))
        state = _state(owner)
        assert fake_offload.log == [] and state.idle and not state.records

    def test_borrowed_tensors_are_not_released(self, fake_offload):
        owner = _Owner()
        release = (TensorRef.input(owner._bda, 1),)
        _install(owner, OffloadStrategy(owner.norm, "g", commit_on=owner._bda, release=release))
        x = torch.randn(2, 16)
        mark_borrowed(x)
        owner(x)
        assert fake_offload.log[-1] == ("commit", "g", ())


class TestResidualReplayPlan:
    def test_block_ends_keep_the_last_layer_output_live(self):
        owners = [_Owner() for _ in range(3)]
        _, strategy = _replay(*owners, block_size=2)
        assert strategy.hooks[0] is strategy  # it acts on the layer step itself
        kinds = [(type(h).__name__, h.region, h.block) for h in strategy.hooks[1:]]
        assert kinds == [
            ("_ReplayHook", owners[0].norm, 0),
            ("_ReplayHook", owners[0]._bda, 0),
            ("_ReplayHook", owners[1].norm, 0),
            ("_ReplayHook", owners[2].norm, 1),
        ]
        assert strategy.block_end_steps == {1: 0, 2: 1}
        assert strategy.managers == [None, None]

    def test_block_cannot_end_inside_a_layer_step(self):
        class _Pair(nn.Module):
            def __init__(self):
                super().__init__()
                self.first, self.second = _Owner(), _Owner()

            def get_residual_branches_by_layer(self):
                return [
                    *self.first.get_residual_branches_by_layer(),
                    *self.second.get_residual_branches_by_layer(),
                ]

        with pytest.raises(ValueError, match="cannot end inside a layer step"):
            install_residual_replay([_Pair()], block_size=1)
