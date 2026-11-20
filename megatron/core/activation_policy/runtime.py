# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run installed activation strategies around the calls of their points.

A module builds its :mod:`~megatron.core.activation_policy.strategies` in ``__init__`` and
installs them with :func:`install_activation_strategies` (via ``MegatronModule``). A strategy
acts on a *point* of its owner: a child module or a method. ``megatron.core.transformer.module``
hooks the points transparently and hands each call here; for every call the runtime runs the
point's strategies around it, records the boundary tensors they capture, and fires the actions
they scheduled on the point's enter and exit events. It knows no recompute or offload specifics,
and model code never calls it.

A strategy may also act on the points of another module (``strategy.owner``): a block-level
strategy contributes hooks to its layers' points, which keep them across their own installs.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Iterator, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.activation_policy.strategies import Event, TensorRef, When

_INSTALLED_ATTR = "_activation_policy"
# Strategies other owners contributed to this module's regions: {id(contributor): [strategy]}.
_CONTRIBUTED_ATTR = "_activation_policy_contributed"


# ---------------------------------------------------------------------------------------------
# Recompute guard
# ---------------------------------------------------------------------------------------------

_RECOMPUTING = 0


@contextmanager
def recompute_guard() -> Iterator[None]:
    """Mark a backward recompute of a region (strategies may skip their forward-only work)."""
    global _RECOMPUTING
    _RECOMPUTING += 1
    try:
        yield
    finally:
        _RECOMPUTING -= 1


def recomputing() -> bool:
    """Whether a region is being recomputed for backward."""
    return _RECOMPUTING > 0


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _element(value: Any, index: int) -> Any:
    if isinstance(value, (tuple, list)):
        return value[index] if index < len(value) else None
    return value if index == 0 else None


# ---------------------------------------------------------------------------------------------
# Per-owner state: the event bus and captured boundary tensors
# ---------------------------------------------------------------------------------------------


class _OwnerState:
    """Scheduled actions and captured boundary tensors of an owner's current forward.

    Actions are keyed by their source, so a stale action from a forward whose event never fired
    (for example an aborted micro-batch) is replaced rather than fired later. Only tensors that a
    strategy captures are recorded, and each is dropped once used, so the state never extends an
    activation's lifetime.
    """

    __slots__ = ("pending", "armed", "records")

    def __init__(self):
        # (region, when) -> {source: action}, in the order the actions were scheduled.
        self.pending: dict[tuple[str, When], dict[Hashable, Any]] = {}
        # References a strategy captures, and the tensors recorded for them.
        self.armed: set[TensorRef] = set()
        self.records: dict[TensorRef, Any] = {}

    @property
    def idle(self) -> bool:
        """Whether nothing is pending (most regions then need no bookkeeping at all)."""
        return not (self.pending or self.armed)

    def defer(self, event: Event, source: Hashable, action) -> None:
        bucket = self.pending.setdefault((event.region, event.when), {})
        # Re-insert at the end: LIFO stages fire in reverse order of scheduling.
        bucket.pop(source, None)
        bucket[source] = action

    def take(self, region: str, when: When) -> list:
        bucket = self.pending.pop((region, when), None)
        return list(bucket.values()) if bucket else []

    def record(self, refs: tuple[TensorRef, ...], side: str, values: Any) -> None:
        for ref in refs:
            if ref.side == side and ref in self.armed:
                self.records[ref] = _element(values, ref.index)

    def take_record(self, ref: TensorRef) -> Any:
        self.armed.discard(ref)
        return self.records.pop(ref, None)


def _fire_enter(state: _OwnerState, region: str, hook_tensors: list) -> None:
    for action in state.take(region, "enter"):
        action.on_enter(hook_tensors)


class _Installed:
    """The strategies installed on an owner, by point, and the owner's state.

    A point is a child module or a method of the owner. Its strategies are the owner's own and
    those other owners contributed.
    """

    __slots__ = ("own", "contributes_to", "strategies", "observed", "refs", "state")

    def __init__(self, own: list, contributed: list, contributes_to: tuple = ()):
        # The owner's own (expanded) strategies, and the modules it contributed strategies to.
        self.own = own
        self.contributes_to = contributes_to
        by_region: dict = {}
        for strategy in own + contributed:
            if isinstance(strategy.region, str):
                raise TypeError(
                    f"{type(strategy).__name__} names point {strategy.region!r}: a point is a "
                    "child module or a method, not a name."
                )
            by_region.setdefault(strategy.region, []).append(strategy)
        self.strategies = {
            region: _strategies.order_region(region, group) for region, group in by_region.items()
        }
        observed = set(self.strategies)
        refs: dict = {}
        for strategies in self.strategies.values():
            for strategy in strategies:
                observed.update(strategy.observes())
                for ref in strategy.refs():
                    refs.setdefault(ref.region, []).append(ref)
        self.observed = frozenset(observed)
        self.refs = {region: tuple(r) for region, r in refs.items()}
        self.state = _OwnerState()


def _installed(owner: torch.nn.Module) -> Optional[_Installed]:
    return owner.__dict__.get(_INSTALLED_ATTR)


# ---------------------------------------------------------------------------------------------
# Region execution
# ---------------------------------------------------------------------------------------------


def _select(owner: torch.nn.Module, strategies) -> tuple[Any, list]:
    """The active wrapping strategy (if any) and the active non-wrapping ones of a call.

    Replaying strategies come first (order 0); the strategies after them see whether the call
    is replayed.
    """
    replayed = False
    wrapper = None
    others = []
    for strategy in strategies:
        if strategy.active(owner, replayed):
            replayed = replayed or strategy.replays
            if not strategy.wraps:
                others.append(strategy)
            elif wrapper is None:
                wrapper = strategy
    return wrapper, others


def _run_region(
    owner: torch.nn.Module,
    installed: _Installed,
    name: str,
    forward: Callable,
    recompute: Callable,
    args: tuple,
    kwargs: dict,
):
    """Run region ``name`` through its strategies (see ``strategies``)."""
    strategies = installed.strategies.get(name, ())
    state = installed.state
    if not strategies and state.idle:
        return forward(*args, **kwargs)
    wrapper, others = _select(owner, strategies)
    if wrapper is None and not others and state.idle:
        return forward(*args, **kwargs)

    call = _strategies.RegionCall(owner, name, state)
    if args and (name, "enter") in state.pending:
        _fire_enter(state, name, [_first_tensor(args[0])])

    entered = []
    try:
        for strategy in others:
            args = strategy.enter(call, args, kwargs)
            entered.append(strategy)
        refs = installed.refs.get(name)
        if refs:
            state.record(refs, "input", args)
        if wrapper is not None:
            out = wrapper.run(call, forward, recompute, args, kwargs)
        else:
            out = forward(*args, **kwargs)
    except BaseException:
        for strategy in entered:
            strategy.abort(call)
        raise
    return _exit_region(installed, call, others, out)


def _exit_region(installed: _Installed, call, strategies, out: Any) -> Any:
    """Everything that happens when a region returns ``out``; returns the final output."""
    for strategy in strategies:
        out = strategy.exit(call, out)
    refs = installed.refs.get(call.name)
    if refs:
        call.state.record(refs, "output", out)
    actions = call.state.take(call.name, "exit")
    return _strategies.fire_exit_actions(actions, out, call.state) if actions else out


# ---------------------------------------------------------------------------------------------
# Public API for model code and executors
# ---------------------------------------------------------------------------------------------


def run_point(owner: torch.nn.Module, point, forward: Callable, recompute: Callable, args, kwargs):
    """Run a call of ``point`` of ``owner`` through its strategies (called by the point hooks)."""
    installed = _installed(owner)
    if installed is None:
        return forward(*args, **kwargs)
    return _run_region(owner, installed, point, forward, recompute, args, kwargs)


def fire_boundary(owner: torch.nn.Module, point, when: When, value: Any, *, inputs: tuple = ()):
    """Fire the enter or exit event of ``point`` without calling it (see
    ``MegatronModule.fire_activation_boundary``).

    ``"enter"``: ``value`` are the tensors that enter-event actions (output releases) hook their
    rebuild on; returns ``None``. ``"exit"``: ``value`` is the point's output, returned after the
    exit-event actions (offload commits) ran on it; ``inputs`` are recorded as the point's
    arguments for pending offload releases. Offload groups do not start at a fired boundary.
    """
    installed = _installed(owner)
    if installed is None or point not in installed.observed:
        return None if when == "enter" else value
    if when == "enter":
        _fire_enter(installed.state, point, list(value))
        return None
    refs = installed.refs.get(point)
    if refs:
        installed.state.record(refs, "input", inputs)
    call = _strategies.RegionCall(owner, point, installed.state)
    _, strategies = _select(owner, installed.strategies.get(point, ()))
    return _exit_region(installed, call, strategies, value)


def has_strategy(owner: torch.nn.Module, point, strategy_type: type) -> bool:
    """Whether ``point`` of ``owner`` has a strategy of ``strategy_type`` installed (for tests and
    diagnostics; model code does not depend on which strategies are installed)."""
    installed = _installed(owner)
    strategies = installed.strategies.get(point, ()) if installed is not None else ()
    return any(isinstance(strategy, strategy_type) for strategy in strategies)


def install_activation_strategies(owner: torch.nn.Module, strategies: Iterable = ()) -> None:
    """Install activation ``strategies`` on ``owner``, replacing any installed before.

    A point is a child module or a method of ``owner`` (``megatron.core.transformer.module``
    hooks it). A strategy may expand into several (see
    :meth:`Strategy.expand`), for example one hook per layer of a block. An expanded strategy
    whose ``owner`` is another module is contributed to that module's regions instead; it stays
    there across that module's own installs until ``owner`` installs again.
    """
    own: list = []
    targets: dict = {}
    for strategy in strategies:
        for expanded in strategy.expand():
            target = getattr(expanded, "owner", None)
            if target is None or target is owner:
                own.append(expanded)
            else:
                targets.setdefault(id(target), (target, []))[1].append(expanded)
    _withdraw(owner)
    _apply(owner, own, tuple(target for target, _ in targets.values()))
    for target, contributed in targets.values():
        target.__dict__.setdefault(_CONTRIBUTED_ATTR, {})[id(owner)] = contributed
        _reapply(target)


def _apply(owner: torch.nn.Module, own: list, contributes_to: tuple = ()) -> None:
    """(Re)build ``owner``'s table from its own and the contributed strategies."""
    _unwrap(owner)
    contributed = [
        strategy
        for group in owner.__dict__.get(_CONTRIBUTED_ATTR, {}).values()
        for strategy in group
    ]
    from megatron.core.transformer.module import attach_activation_point

    installed = _Installed(own, contributed, contributes_to)
    owner.__dict__[_INSTALLED_ATTR] = installed
    for point in installed.observed:
        attach_activation_point(owner, point)


def _reapply(owner: torch.nn.Module) -> None:
    installed = _installed(owner)
    if installed is None:
        _apply(owner, [])
    else:
        _apply(owner, installed.own, installed.contributes_to)


def _withdraw(owner: torch.nn.Module) -> None:
    """Remove the strategies ``owner`` contributed to other modules."""
    installed = _installed(owner)
    for target in installed.contributes_to if installed is not None else ():
        target.__dict__.get(_CONTRIBUTED_ATTR, {}).pop(id(owner), None)
        _reapply(target)


def _unwrap(owner: torch.nn.Module) -> None:
    from megatron.core.transformer.module import detach_activation_point

    installed = owner.__dict__.pop(_INSTALLED_ATTR, None)
    if installed is None:
        return
    for point in installed.observed:
        detach_activation_point(owner, point)


def _uninstall(owner: torch.nn.Module) -> None:
    _withdraw(owner)
    _unwrap(owner)


def installed_strategies(owner: torch.nn.Module) -> dict:
    """The strategies installed on ``owner``, by region, in the order they apply."""
    installed = _installed(owner)
    return dict(installed.strategies) if installed is not None else {}


# Strategies import this module; bind theirs last (attributes are read at call time).
import megatron.core.activation_policy.strategies as _strategies  # noqa: E402
