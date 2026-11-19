# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Megatron Module."""

import copy
import inspect
import weakref
from contextlib import nullcontext
from typing import Any, Optional, Tuple

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    ensure_metadata_has_dp_cp_group,
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)

_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)


class TwoStageAttentionLayer:
    """Interface for attention-like modules that expose core and post-core stages."""

    def supports_two_stage_attention(self) -> bool:
        """Return whether this module instance supports two-stage execution."""
        return True

    def forward_pre_attn_and_core_attn(
        self, *args: Any, packed_sequence_cp_metadata: Any = None, **kwargs: Any
    ) -> Any:
        """Run the pre-attention and core-attention stage."""
        raise NotImplementedError

    def forward_post_core_attn(self, *args: Any, **kwargs: Any) -> Any:
        """Run the post-core-attention stage."""
        raise NotImplementedError


def param_is_not_shared(param):  # pylint: disable=missing-function-docstring
    return not hasattr(param, 'shared') or not param.shared


def is_first_microbatch_tracked(config) -> bool:
    """True if ``is_first_microbatch`` is still being kept up to date.

    A training step runs N microbatches. The flag marks microbatch 1 -- the one that
    re-quantizes the weights and starts a fresh main_grad, while 2..N reuse and accumulate::

        layer is built      ->  flag = True
        every forward       ->  flag = False   (microbatch 1 is over)
        start of each step  ->  flag = True    (only quantized configs)
    """
    return (
        config.fp8 is not None
        or config.fp4 is not None
        or getattr(config, 'use_kitchen', False)
        or getattr(config, 'quant_recipe', None) is not None
    )


class MegatronModule(torch.nn.Module):
    """Base Megatron module inhertied by all Models.

    Megatron specific extensions of torch Module with support
    for pipelining

    Args:
        config (TransformerConfig): Transformer config
    """

    # def __init__(self, config: TransformerConfig, share_word_embeddings=True):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    @classmethod
    def estimate_flops(
        cls,
        config: TransformerConfig,
        *,
        total_real_tokens_in_batch: float,
        seqlen_squared_sum_in_batch: float,
    ) -> float:
        """Estimate this module's forward FLOPs for one global batch from its config.

        Built-in and custom layers share this optional hook. It must be pure:
        do not mutate the config, construct modules, or run collectives.
        Return a finite nonnegative real number. Training accounts for repeated
        layer executions and applies the forward/backward multiplier separately.

        Include multiply/add operations, but exclude model-level work such as logits.
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement estimate_flops for layer FLOP accounting."
        )

    def install_activation_strategies(self, strategies=()) -> None:
        """Install activation strategies on this module, replacing any installed before.

        Modules build their strategies (``megatron.core.activation_policy.strategies``) at the
        end of ``__init__`` and install them here. A strategy acts on a *point*: a child module
        (``self.linear_fc1``) or a method (``self._mlp_bda``). The point is hooked transparently
        (see :class:`_ActivationPointHook`): callers keep calling the same module or method. A
        subclass whose strategies differ installs again at the end of its own ``__init__``,
        replacing the base class's.
        """
        from megatron.core.activation_policy.runtime import install_activation_strategies

        install_activation_strategies(self, strategies)

    def fire_activation_boundary(self, point, when: str, value, *, inputs: tuple = ()):
        """Signal that ``point`` of this module was entered (``when="enter"``, ``value`` the
        tensors to hook on) or returned (``when="exit"``, ``value`` its output, which is
        returned, possibly replaced) without calling it.

        For executors that run this module's stages through another schedule than its own
        calls (for example a CUDA graph that ends where ``point`` would start).
        """
        from megatron.core.activation_policy.runtime import fire_boundary

        return fire_boundary(self, point, when, value, inputs=inputs)

    def get_residual_branches(self) -> list:
        """This layer's residual branches (``activation_policy.ResidualBranch``): the residual
        reads, writes and the ops between them that residual replay hooks. None by default."""
        return []

    def get_residual_branches_by_layer(self) -> list:
        """:meth:`get_residual_branches` per physical layer this module runs (one by default)."""
        return [self.get_residual_branches()]

    def refresh_cache(self) -> None:
        """Refresh state derived from parameters after an in-place weight refit.

        Refit bypasses the normal checkpoint-load and train/eval lifecycles. Modules
        that cache values derived from parameters can override this method; the refit
        receiver calls it after all parameter and buffer transfers have completed.
        """

    def state_dict_for_save_checkpoint(self, prefix: str = '', keep_vars: bool = False):
        """Override state dict for saving checkpoints Use this function to override the
        state dict for saving checkpoints.

        Args:
            prefix (str, optional): _description_. Defaults to ''.
            keep_vars (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        return self.state_dict(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Default implementation for sharded state dict for distributed checkpointing.

        General definition of sharded_state_dict simply calls `sharded_state_dict_default`
        (which call sharded_state_dict method if possible or a default implementation otherwise)
        recursively on all submodules.

        Args:
            prefix (str): prefix for the state dict keys
            sharded_offsets (Tuple[Tuple[int, int, int]], optional): sharding already
                applied (e.g. PP related) by sup-modules. Passed along to ShardedTensor
            metadata (dict, optional): metadata passed recursively to sharded_state_dict methods

        Returns:
            dict: dictionary of state dict keys mapped to ShardedTensors
        """
        sharded_state_dict = {}
        # Save parameters
        self._save_to_state_dict(sharded_state_dict, '', keep_vars=True)
        if not hasattr(self, 'tp_group'):
            # some model interface hasn't updated for m4, fallback needed
            tp_group = parallel_state.get_tensor_model_parallel_group()
        else:
            tp_group = self.tp_group
        # Guard for cases metadata is not provided
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict,
            prefix,
            sharded_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata['dp_cp_group'],
        )
        # Recurse into submodules
        for name, module in self.named_children():
            sharded_state_dict.update(
                sharded_state_dict_default(
                    module, f'{prefix}{name}.', sharded_offsets, metadata, tp_group=tp_group
                )
            )
        return sharded_state_dict

    def set_is_first_microbatch(self):
        """Sets the is_first_microbatch flag if it exists and config.fp8==True.
        When this flag is set, TE modules will update their fp8 parameter cache.
        If kitchen is being used, kitchen controls quantization level.
        A quant_recipe (e.g. from --te-precision-config-file) also enables the flag.
        """
        if is_first_microbatch_tracked(self.config):
            if not hasattr(self, "modules_with_is_first_microbatch"):
                self.modules_with_is_first_microbatch = []
                for m in self.modules():
                    if hasattr(m, "is_first_microbatch"):
                        self.modules_with_is_first_microbatch.append(m)
            for m in self.modules_with_is_first_microbatch:
                m.is_first_microbatch = True

    def set_symmetric_ar(self, set_to: Optional[str] = None) -> None:
        """
        Set symmetric all-reduce functionality across all eligible modules.

        This method traverses the model's module hierarchy to find all modules
        with the 'symmetric_ar_type' attribute, caches them, and then sets their
        '_symmetric_ar_cache' attribute to the specified value to enable or
        disable symmetric all-reduce operations.

        Args:
            set_to (Any, optional): Value to set for the 'symmetric_ar_type' to.
            Allowed choices ['two_shot', "one_shot", "multimem_all_reduce", None]
        """
        assert set_to in ['two_shot', "one_shot", "multimem_all_reduce", None]

        # Recursive function to find all modules with our target attributes
        def create_ar_cache(module):
            # Check if this module has any of our target attributes
            if hasattr(module, "symmetric_ar_type"):
                self._symmetric_ar_cache.append(module)

            # Check all children modules recursively
            for child in module._modules.values():
                if child is not None:
                    create_ar_cache(child)

        if not hasattr(self, "_symmetric_ar_cache"):
            self._symmetric_ar_cache = []
            create_ar_cache(self)

        for module in self._symmetric_ar_cache:
            module._symmetric_ar_cache = set_to


class GraphableMegatronModule(MegatronModule):
    """Megatron module that can capture and replay local CUDA graphs.

    Args:
        config (TransformerConfig): Transformer config
    """

    def __init__(self, config: TransformerConfig, vp_stage: Optional[int] = None):
        super().__init__(config)

        assert isinstance(config, TransformerConfig), "config must be a TransformerConfig"

        # Enable cuda graphs.
        if config.cuda_graph_impl == "local":
            if hasattr(self, "create_mcore_cudagraph_manager"):
                self.create_mcore_cudagraph_manager(config)
            else:
                from megatron.core.transformer.cuda_graphs import CudaGraphManager

                self.cudagraph_manager = CudaGraphManager(config)

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        return hasattr(self, 'cudagraph_manager')

    def get_inner_quantization_context(self):
        """This module's own FP8/FP4 context, entered around every call (layers with a
        per-layer recipe override it; none by default)."""
        return nullcontext()

    def __call__(self, *args, **kwargs):
        with self.get_inner_quantization_context():
            if self._should_call_local_cudagraph(*args, **kwargs):
                return self.cudagraph_manager(self, args, kwargs)
            return super().__call__(*args, **kwargs)


def conversion_helper(val, conversion):
    """Recursively applies a conversion function to values in nested data structures.

    Args:
        val: A single value or a nested structure (tuple/list) of values to convert
        conversion (callable): A function that performs the desired conversion on a single value

    Returns:
        The converted value, maintaining the same nested structure as the input.
        If input is a single value, returns the converted value.
        If input is a tuple/list, returns a tuple/list with all elements converted.
    """
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Converts floating-point values from fp32 to fp16.

    Args:
        val: The value to convert. Can be a single number, a tuple, or a list.
        float16_convertor: A function that converts a single fp32 value to fp16
    """

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Converts floating-point values from fp16 to fp32.

    Args:
        val: The value to convert. Can be a single number, a tuple, or a list.
    """

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


def mark_keep_in_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Mark a parameter or buffer so that ``Float16Module`` keeps it in FP32.

    Args:
        tensor: The parameter or buffer to mark.

    Returns:
        The same tensor, for call-site convenience.
    """
    tensor.keep_in_fp32 = True
    return tensor


def convert_module_to_dtype_except_fp32_marked(
    module: torch.nn.Module, dtype: torch.dtype
) -> torch.nn.Module:
    """Cast floating-point parameters and buffers except those marked to stay in FP32.

    Args:
        module: The module to convert in place.
        dtype: The target floating-point dtype.

    Returns:
        The converted module.
    """
    return module._apply(
        lambda tensor: (
            tensor.to(dtype)
            if tensor.is_floating_point() and not getattr(tensor, 'keep_in_fp32', False)
            else tensor
        )
    )


class Float16Module(MegatronModule):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(Float16Module, self).__init__(config)
        self.config = config
        self.fp16 = config.fp16
        self.bf16 = config.bf16
        self.vp_size = config.virtual_pipeline_model_parallel_size
        self.vp_stage = getattr(module, 'vp_stage', None)
        self.pg_collection = getattr(module, 'pg_collection', None)

        if self.fp16:
            self.add_module(
                'module', convert_module_to_dtype_except_fp32_marked(module, torch.half)
            )

            def float16_convertor(val):
                return val.half()

        elif self.bf16:
            self.add_module(
                'module', convert_module_to_dtype_except_fp32_marked(module, torch.bfloat16)
            )

            def float16_convertor(val):
                return val.bfloat16()

        else:
            raise Exception('Either config.fp16 or config.bf16 should be True.')

        self.float16_convertor = float16_convertor

    def set_input_tensor(self, input_tensor):  # pylint: disable=missing-function-docstring
        return self.module.set_input_tensor(input_tensor)

    def forward(self, *inputs, fp32_output=True, **kwargs):
        """
        Execute the wrapped module in model precision and optionally upcast outputs to fp32.

        On the first pipeline stage, positional/keyword tensor inputs are converted to the
        module precision (fp16 or bf16) before invoking the wrapped module. The wrapped module
        is called with the provided inputs and keyword arguments. On the last pipeline stage
        only, outputs are upcast to fp32 if ``fp32_output`` is True; otherwise, outputs are
        returned in the model precision (fp16/bf16).

        Args:
            *inputs: Positional inputs forwarded to the wrapped module (converted to fp16/bf16 on
                the pipeline first stage).
            fp32_output (bool, keyword-only): If True (default), upcast outputs to fp32 on the
                pipeline last stage. Has no effect on non-last stages. Set to False to keep outputs
                in model precision when downstream consumers expect half precision or to avoid
                extra casts.
            **kwargs: Keyword arguments forwarded to the wrapped module.

        Returns:
            The wrapped module's outputs, potentially upcast to fp32 depending on pipeline stage
            and ``fp32_output``.
        """
        from megatron.core.pipeline_parallel.utils import (
            is_pp_first_stage,
            is_pp_last_stage,
            is_vp_first_stage,
            is_vp_last_stage,
        )

        if self.pg_collection is None:
            pp_group = parallel_state.get_pipeline_model_parallel_group()
        else:
            pp_group = self.pg_collection.pp
        if is_vp_first_stage(self.vp_stage, self.vp_size) and is_pp_first_stage(pp_group):
            inputs = fp32_to_float16(inputs, self.float16_convertor)
        outputs = self.module(*inputs, **kwargs)
        if (
            is_vp_last_stage(self.vp_stage, self.vp_size)
            and is_pp_last_stage(pp_group)
            and fp32_output is True
        ):
            outputs = float16_to_fp32(outputs)
        return outputs

    def state_dict(
        self, destination=None, prefix='', keep_vars=False
    ):  # pylint: disable=missing-function-docstring
        return self.module.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Retrieve state_dict from the module being wrapped."""
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def sharded_state_dict(self, prefix='', *args, **kwargs):
        """Retrieve sharded_state_dict from the module being wrapped."""
        return self.module.sharded_state_dict(prefix, *args, **kwargs)

    def load_state_dict(
        self, state_dict, strict=True
    ):  # pylint: disable=missing-function-docstring
        self.module.load_state_dict(state_dict, strict=strict)


# ---------------------------------------------------------------------------------------------
# Activation-policy hooks
# ---------------------------------------------------------------------------------------------
#
# A strategy installed on an owner acts on a *point*: a child module or a method. The point is
# hooked here, transparently: the module object, its class and how callers call it are
# unchanged. A hooked call is handed to ``megatron.core.activation_policy.runtime``, which runs
# the owner's strategies around it; strategies never see how they are attached.


class _ActivationPointHook:
    """Base of the per-point hooks: routes a call of the point to the owner's strategies.

    The owner is held weakly: a recomputed call keeps itself alive from autograd hooks (which the
    garbage collector cannot traverse) until backward. If that backward never happens, e.g. for
    a tensor owned by a CUDA graph, a strong reference would pin the whole owner.
    """

    def _dispatch(self, point, forward, recompute, args, kwargs):
        owner = self._owner()
        if owner is None:
            return forward(*args, **kwargs)
        from megatron.core.activation_policy.runtime import run_point

        return run_point(owner, point, forward, recompute, args, kwargs)

    @property
    def owner(self):
        """The owner module (``None`` once it has been freed)."""
        return self._owner()


class _ModulePointHook(_ActivationPointHook):
    """Instance-level ``forward`` of a child module hooked by its owner.

    It runs inside ``nn.Module.__call__``, so the child's own hooks (parameter gathering, CUDA
    graphs) wrap the hooked call as before. A recompute re-enters through ``__call__`` so those
    hooks run again.
    """

    def __init__(self, owner, module: torch.nn.Module):
        self._owner = weakref.ref(owner)
        self.module, self.forward = module, module.forward
        # An instance-level ``forward`` the module had before (restored on removal).
        self.instance_forward = module.__dict__.get("forward")
        self.bypass = False

    def __deepcopy__(self, memo):
        # When the owner is copied along with this child, route the copy through the owner's
        # copy. A child copied on its own gets a plain forward: its copy has no owner.
        owner = self.owner
        owner_copy = memo.get(id(owner)) if owner is not None else None
        if owner_copy is None:
            return copy.deepcopy(self.forward, memo)
        clone = _ModulePointHook.__new__(_ModulePointHook)
        memo[id(self)] = clone
        clone._owner = weakref.ref(owner_copy)
        clone.module = copy.deepcopy(self.module, memo)
        clone.forward = copy.deepcopy(self.forward, memo)
        clone.instance_forward = copy.deepcopy(self.instance_forward, memo)
        clone.bypass = False
        return clone

    def _recompute(self, *args, **kwargs):
        self.bypass = True
        try:
            return self.module(*args, **kwargs)
        finally:
            self.bypass = False

    def __call__(self, *args, **kwargs):
        if self.bypass:
            return self.forward(*args, **kwargs)
        return self._dispatch(self.module, self.forward, self._recompute, args, kwargs)


class _MethodPointHook(_ActivationPointHook):
    """Instance attribute shadowing a hooked method of its owner (``self.m(...)`` still works).

    It compares and hashes equal to the bound method it shadows, so a point named as ``self.m``
    before or after the hook is attached is the same point.
    """

    def __init__(self, owner, method):
        self._owner = weakref.ref(owner)
        if isinstance(method, _MethodPointHook):  # named after its hook was attached
            self.func, self.name = method.func, method.name
        else:
            self.func, self.name = method.__func__, method.__name__

    def _bound(self):
        return self.func.__get__(self._owner())

    def __eq__(self, other):
        if isinstance(other, _MethodPointHook):
            return self.func is other.func and self._owner() is other._owner()
        if inspect.ismethod(other):
            return other.__func__ is self.func and other.__self__ is self._owner()
        return NotImplemented

    def __hash__(self):
        return hash(self._bound())

    def __deepcopy__(self, memo):
        owner = self.owner
        clone = _MethodPointHook.__new__(_MethodPointHook)
        memo[id(self)] = clone
        owner_copy = memo.get(id(owner)) if owner is not None else None
        clone._owner = weakref.ref(owner_copy) if owner_copy is not None else lambda: None
        clone.func, clone.name = self.func, self.name
        return clone

    def __call__(self, *args, **kwargs):
        if self._owner() is None:
            raise RuntimeError(f"The owner of hooked method {self.name!r} was freed.")
        method = self._bound()
        return self._dispatch(method, method, method, args, kwargs)


def is_method_point(point) -> bool:
    """Whether ``point`` is a method (bound, or already hooked) rather than a child module."""
    return inspect.ismethod(point) or isinstance(point, _MethodPointHook)


def attach_activation_point(owner: torch.nn.Module, point) -> None:
    """Hook ``point`` (a child module, or a method of ``owner``) for ``owner``'s strategies."""
    if is_method_point(point):
        bound_owner = point.__self__ if inspect.ismethod(point) else point.owner
        if bound_owner is not owner:
            raise ValueError(
                f"Method point {point.__name__ if inspect.ismethod(point) else point.name!r} "
                f"must be a method of its owner {type(owner).__name__}."
            )
        name = point.__name__ if inspect.ismethod(point) else point.name
        if not isinstance(owner.__dict__.get(name), _MethodPointHook):
            owner.__dict__[name] = _MethodPointHook(owner, point)
        return
    if not isinstance(point, torch.nn.Module):
        raise TypeError(
            f"An activation point is a child module or a method, got {type(point).__name__}."
        )
    if point is owner:
        raise ValueError("A module cannot be a point of itself.")
    hook = point.__dict__.get("forward")
    if isinstance(hook, _ModulePointHook):
        if hook.owner is owner:
            return
        raise RuntimeError(f"Module {type(point).__name__} is already hooked by another owner.")
    point.forward = _ModulePointHook(owner, point)


def detach_activation_point(owner: torch.nn.Module, point) -> None:
    """Remove the hook :func:`attach_activation_point` put on ``point`` for ``owner``."""
    if is_method_point(point):
        name = point.__name__ if inspect.ismethod(point) else point.name
        if isinstance(owner.__dict__.get(name), _MethodPointHook):
            del owner.__dict__[name]
        return
    if not isinstance(point, torch.nn.Module):
        return
    hook = point.__dict__.get("forward")
    if isinstance(hook, _ModulePointHook) and hook.owner is owner:
        if hook.instance_forward is None:
            del point.__dict__["forward"]
        else:
            point.forward = hook.instance_forward
