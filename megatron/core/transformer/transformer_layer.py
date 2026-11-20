# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import functools
import logging
import warnings
from abc import ABC
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Union

import torch
import torch.distributed
from torch import Tensor

try:
    from nemo.lens.helpers import managed_span as _otel_managed_span
except ImportError:
    from megatron.core.telemetry.fallbacks import managed_span as _otel_managed_span

from megatron.core import parallel_state
from megatron.core.activation_policy import (
    CheckpointStrategy,
    CheckpointWithoutOutputStrategy,
    Event,
    ExcludeFromOffloadStrategy,
    OffloadStrategy,
    TensorRef,
)
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.dist_checkpointing.utils import apply_prefix_mapping
from megatron.core.enums import Fp8Recipe
from megatron.core.inference.utils import InferenceMode
from megatron.core.ops.norm.torch_norm import LayerNormBuilder
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope, LayerType
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp, is_identity
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.module import GraphableMegatronModule, TwoStageAttentionLayer
from megatron.core.transformer.moe.router_replay import ChunkedRouterReplay
from megatron.core.transformer.residual_connection import (
    ResidualBranchOutput,
    ResidualConnection,
    ResidualConnectionOperation,
    ResidualConnectionState,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer_mixins import TECudaGraphTransformerLayerMixin
from megatron.core.typed_torch import apply_module, copy_signature
from megatron.core.utils import (
    deprecate_inference_params,
    get_pg_rank,
    is_te_min_version,
    is_torch_min_version,
    log_single_rank,
    make_viewless_tensor,
    nvtx_range_pop,
    nvtx_range_push,
)

if TYPE_CHECKING:
    from megatron.core.inference.contexts import BaseInferenceContext

logger = logging.getLogger(__name__)


def _get_offloading_interface():
    """Get the offloading interface for fine-grained activation offloading."""
    # Keep this import lazy to avoid a transformer/pipeline circular import.
    from megatron.core.activation_policy.offload.manager import (
        FineGrainedActivationOffloadingInterface,
    )

    return FineGrainedActivationOffloadingInterface


def get_transformer_layer_offset(
    config: TransformerConfig, vp_stage: Optional[int] = None, pp_rank: Optional[int] = None
):
    """Get the index offset of current pipeline stage, given the level of pipelining."""
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

    is_first_pp_stage = pp_rank == 0

    if config.pipeline_model_parallel_size > 1:

        if config.pipeline_model_parallel_layout:
            offset = config.pipeline_model_parallel_layout.get_layer_offset(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        elif (
            config.num_layers_in_first_pipeline_stage is not None
            or config.num_layers_in_last_pipeline_stage is not None
        ):
            # Calculate number of pipeline stages to distribute the remaining Transformer
            # layers after deducting the Transformer layers in the first or the last stages
            middle_pipeline_stages = config.pipeline_model_parallel_size
            middle_pipeline_stages -= sum(
                [
                    1 if x is not None else 0
                    for x in (
                        config.num_layers_in_first_pipeline_stage,
                        config.num_layers_in_last_pipeline_stage,
                    )
                ]
            )

            # Calculate layers to distribute in each pipeline stage. If the
            # num_layers_in_first_pipeline_stage and num_layers_in_last_pipeline_stage
            # are not set, we will not enable uneven pipeline. All layers will be treated
            # as middle layers.
            num_layers_in_first_pipeline_stage = (
                0
                if config.num_layers_in_first_pipeline_stage is None
                else config.num_layers_in_first_pipeline_stage
            )
            num_layers_in_last_pipeline_stage = (
                0
                if config.num_layers_in_last_pipeline_stage is None
                else config.num_layers_in_last_pipeline_stage
            )

            middle_num_layers = (
                config.num_layers
                - num_layers_in_first_pipeline_stage
                - num_layers_in_last_pipeline_stage
            )

            middle_pipeline_rank = (
                pp_rank if config.num_layers_in_first_pipeline_stage is None else pp_rank - 1
            )

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                # Calculate number of layers in each virtual model chunk
                # If the num_layers_in_first_pipeline_stage and
                # num_layers_in_last_pipeline_stage are not set, all pipeline stages
                # will be treated as middle pipeline stages in the calculation
                num_layers_per_virtual_model_chunk_in_first_pipeline_stage = (
                    0
                    if config.num_layers_in_first_pipeline_stage is None
                    else config.num_layers_in_first_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_last_pipeline_stage = (
                    0
                    if config.num_layers_in_last_pipeline_stage is None
                    else config.num_layers_in_last_pipeline_stage // vp_size
                )

                num_layers_per_virtual_model_chunk_in_middle_pipeline_stage = (
                    middle_num_layers // vp_size
                )

                # First stage + middle stage + last stage
                total_virtual_chunks = (
                    num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_middle_pipeline_stage
                    + num_layers_per_virtual_model_chunk_in_last_pipeline_stage
                )

                # Calculate the layer offset with interleaved uneven pipeline parallelism
                if pp_rank == 0:
                    offset = vp_stage * total_virtual_chunks
                else:
                    offset = (
                        vp_stage * total_virtual_chunks
                        + num_layers_per_virtual_model_chunk_in_first_pipeline_stage
                        + middle_pipeline_rank
                        * (
                            num_layers_per_virtual_model_chunk_in_middle_pipeline_stage
                            // middle_pipeline_stages
                        )
                    )
            else:
                if middle_pipeline_stages > 0:
                    num_layers_per_pipeline_rank = middle_num_layers // middle_pipeline_stages
                else:
                    num_layers_per_pipeline_rank = 0

                if pp_rank == 0:
                    offset = 0
                else:
                    offset = (
                        middle_pipeline_rank * num_layers_per_pipeline_rank
                    ) + num_layers_in_first_pipeline_stage
        else:
            num_layers = config.num_layers

            # Increase the number of layers by one if we include the embedding (loss)
            # layer into pipeline parallelism partition and placement
            if config.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if config.account_for_loss_in_pipeline_split:
                num_layers += 1

            num_layers_per_pipeline_rank = num_layers // config.pipeline_model_parallel_size

            # import here to avoid circular import
            from megatron.core.pipeline_parallel.utils import is_vp_first_stage

            if (vp_size := config.virtual_pipeline_model_parallel_size) is not None:
                assert (
                    vp_stage is not None
                ), "vp_stage must be provided if virtual pipeline model parallel size is set"

                num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
                total_virtual_chunks = num_layers // vp_size
                offset = vp_stage * total_virtual_chunks + (pp_rank * num_layers_per_virtual_rank)

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split and not (
                    is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage
                ):
                    offset -= 1
            else:
                offset = pp_rank * num_layers_per_pipeline_rank

                # Reduce the offset of embedding layer from the total layer number
                if config.account_for_embedding_in_pipeline_split and not (
                    is_vp_first_stage(vp_stage, vp_size) and is_first_pp_stage
                ):
                    offset -= 1
    else:
        offset = 0
    return offset


class MlpInterface(Protocol):
    """Interface for MLP implementations in the transformer layer."""

    def forward(
        self,
        hidden_states: Tensor,
        /,
        *,
        intermediate_tensors: tuple[Tensor, ...] | None = None,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward method for the MLP interface."""
        ...


class MlpBuilder(Protocol):
    """MLP builder protocol for building MLPs in the transformer layer."""

    def __call__(
        self,
        *,
        config: TransformerConfig,
        pg_collection: ProcessGroupCollection,
        is_mtp_layer: bool,
        name: str | None = None,
        hash_moe_layer_threshold: int | None = None,
    ) -> MlpInterface: ...


@dataclass
class TransformerLayerSubmodules:
    """
    Configuration class for specifying the submodules of a transformer layer.

    This class defines the structure and default implementations for various
    components of a transformer layer, allowing for flexible customization
    of the layer's architecture.

    Args:
        input_layernorm: Specification for the input layer normalization.
        self_attention (Union[ModuleSpec, type]): Specification for the self-attention mechanism.
        self_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after self-attention.
        pre_cross_attn_layernorm: Specification for the layer
            normalization before cross-attention.
        cross_attention (Union[ModuleSpec, type]): Specification for the cross-attention mechanism.
        cross_attn_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after cross-attention.
        pre_mlp_layernorm: Specification for the layer normalization
            before the MLP.
        mlp (Union[ModuleSpec, type]): Specification for the MLP in Dense layer.
        mlp_bda (Union[ModuleSpec, type]): Specification for the bias-dropout-add operation
            after the MLP.
        sharded_state_dict_keys_map (Dict[str, str]): Mapping for sharded tensor keys to be applied
            in the `sharded_state_dict` method.
    """

    input_layernorm: LayerNormBuilder = IdentityOp
    self_attention_hyper_connection: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: LayerNormBuilder = IdentityOp
    cross_attention_hyper_connection: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: LayerNormBuilder = IdentityOp
    mlp_hyper_connection: Union[ModuleSpec, type] = IdentityOp
    mlp: MlpBuilder | type[IdentityOp] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)


def _set_offload_in_cuda_graph(layer) -> None:
    """Whether the layer's offloaded modules are captured in its CUDA graph."""
    config = layer.config
    mlp_norm = config.offload_selected("mlp_norm") and not is_identity(layer.pre_mlp_layernorm)
    cuda_graph_modules = config.cuda_graph_modules or []
    attn_regions = any(config.offload_selected(m) for m in ("core_attn", "attn_proj", "qkv_linear"))
    in_graph = (CudaGraphModule.attn in cuda_graph_modules and attn_regions) or (
        not layer.is_moe_layer and CudaGraphModule.mlp in cuda_graph_modules and mlp_norm
    )
    layer.offload_module_in_cuda_graph = in_graph
    if in_graph:
        assert is_torch_min_version(
            "2.9.0a0"
        ), "Offloading modules captured in cuda graph requires torch>=2.9.0."
        assert is_te_min_version(
            "2.14.0"
        ), "Offloading modules captured in cuda graph requires TE>=2.14.0."
        assert (
            config.cuda_graph_warmup_steps > 0
        ), "Fine-grained activation offloading needs cuda_graph_warmup_steps > 0."


class BaseTransformerLayer(ABC):
    """A common parent class for `TransformerLayer` like implementations.

    A dummy class that is subclassed by similar `TransformerLayer`s e.g. the
    `TransformerLayer` in this file and possibly other `TransformerLayer`
    implementations that aim to use `TransformerBlock` as the base module.
    The main purpose is to check if any layer (or module) provided in the spec
    is a subclass of this class to allow fanning-out of that spec for all the
    layers in the `TransformerBlock`. See `_get_block_submodules` method
    implementation in `transformer_block.py` file for more details.
    """

    #: Whether the layer accepts mHC n-stream hidden states ([s, b, n*C])
    #: when owned directly by a TransformerBlock. Custom layer implementations
    #: that implement this contract should override the marker with ``True``.
    supports_mhc_connections: bool = False

    #: Whether the layer owns wide-residual branch connections.
    supports_wide_residual_connections: bool = False

    def __init__(self):
        pass


class _TransformerLayer(GraphableMegatronModule, BaseTransformerLayer, TwoStageAttentionLayer):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.

    The layer only defines the computation. Activation recompute and offloading of its regions
    (``input_layernorm``, ``self_attn_bda``, ``pre_mlp_layernorm``, ``mlp``, ``mlp_bda``) are
    decided by the activation strategies the layer installs in ``__init__`` and applied around
    those calls by :mod:`megatron.core.activation_policy`.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
        hash_moe_layer_threshold: Optional[int] = None,
        name: str | None = None,
    ):
        """
        Args:
            name (str | None): module instance name passed top-down from its paranet module
        """
        self.submodules_config = submodules
        # GraphableMegatronModule may create a local CUDA graph manager during super().__init__().
        # Dense layers need a default before that hook runs, while MoETransformerLayer sets this
        # to True before entering this constructor.
        self.is_moe_layer = getattr(self, "is_moe_layer", False)
        super().__init__(config=config, vp_stage=vp_stage)

        if (
            config.wide_residual is not None
            and not self.supports_wide_residual_connections
            and not is_mtp_layer
        ):
            raise ValueError(
                f"{type(self).__name__} does not implement wide-residual streams. Build the "
                "decoder with WideResidualTransformerLayer when wide_residual is configured."
            )

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp

        # MTP inner layers use their own layer numbering (starting from 1 within each MTP depth),
        # so they should NOT add the decoder layer offset. The router.py handles MTP layer
        # numbering separately by adding config.num_layers to distinguish MTP layers from decoder
        # layers in the aux loss tracker.
        #
        # When add_layer_offset is False, the caller has already included the correct offset
        # in layer_number (e.g. when using --hybrid-layer-pattern with fVPP).
        if is_mtp_layer or not add_layer_offset:
            self.layer_number = layer_number
        else:
            self.layer_number = layer_number + get_transformer_layer_offset(
                self.config, vp_stage, get_pg_rank(pg_collection.pp)
            )
        self.hidden_dropout = config.hidden_dropout if hidden_dropout is None else hidden_dropout
        self.is_mtp_layer = is_mtp_layer

        # [Module 1: Input Layernorm] Optional Layernorm on the input data
        # TODO: add pytorch only layernorm
        self.input_layernorm = submodules.input_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self._input_layernorm_returns_residual = getattr(
            self.input_layernorm, 'returns_residual', False
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                # layer_number is 1-indexed, so we need to subtract 1 to get the correct index
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[
                    self.layer_number - 1
                ]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type

        attention_optional_kwargs["pg_collection"] = pg_collection
        if pp_layer_offset is not None:
            attention_optional_kwargs["pp_layer_offset"] = pp_layer_offset
        attention_optional_kwargs["is_mtp_layer"] = is_mtp_layer

        # [Module 2: SelfAttention]
        self.self_attention = build_module(
            submodules.self_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
            name=(name + ".self_attention") if name is not None else None,
        )

        # [Module 3: BiasDropoutFusion]
        self.self_attn_bda = build_module(submodules.self_attn_bda)

        # [Module 4: Post SelfAttention] Optional Layernorm after self-attn
        self.pre_cross_attn_layernorm = submodules.pre_cross_attn_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self._pre_cross_attn_layernorm_returns_residual = getattr(
            self.pre_cross_attn_layernorm, 'returns_residual', False
        )

        # [Module 5: CrossAttention]
        self.cross_attention = build_module(
            submodules.cross_attention,
            config=self.config,
            layer_number=self.layer_number,
            **attention_optional_kwargs,
            name=(name + ".cross_attention") if name is not None else None,
        )

        # [Module 6: BiasDropoutFusion]
        self.cross_attn_bda = build_module(submodules.cross_attn_bda, config=self.config)

        # [Module 7: Pre MLP] Optional Layernorm before MLP
        self.pre_mlp_layernorm = submodules.pre_mlp_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self._pre_mlp_layernorm_returns_residual = getattr(
            self.pre_mlp_layernorm, 'returns_residual', False
        )
        # [Module 8: MLP block]
        # import here to avoid circular import
        from megatron.core.extensions.transformer_engine import TEFusedMLP
        from megatron.core.transformer.moe.moe_layer import MoELayer

        # MLP expects tp_group but MoELayer expects pg_collection to be passed in.
        # We can change MLP to accept pg_collection but it makes the logic implicit
        # The conditional below is to make the logic explicit
        # if submodules.mlp is not a ModuleSpec,we dont have to handle passing additional kwargs
        if isinstance(submodules.mlp, ModuleSpec) and submodules.mlp.module in (MLP, TEFusedMLP):
            submodules.mlp = functools.partial(
                submodules.mlp.module.as_mlp_submodule,
                submodules=submodules.mlp.submodules,
                **submodules.mlp.params,
            )
            log_single_rank(
                logger,
                logging.WARNING,
                f"Rewrapping ModuleSpec with module {type(submodules.mlp)} to forward kwargs. "
                "Consider migrating the `mlp` submodule spec to a direct call of the "
                "`as_mlp_submodule` classmethod instead.",
            )
        self.mlp = submodules.mlp(
            config=self.config,
            pg_collection=pg_collection,
            is_mtp_layer=self.is_mtp_layer,
            name=(name + ".mlp") if name is not None else None,
            hash_moe_layer_threshold=hash_moe_layer_threshold,
        )
        if hasattr(self.mlp, 'set_layer_number'):
            self.mlp.set_layer_number(self.layer_number)

        # [Module 9: BiasDropoutFusion]
        self.mlp_bda = build_module(submodules.mlp_bda)

        self.is_moe_layer = isinstance(self.mlp, MoELayer)

        # Used by the CUDA-graph capture/replay paths to order offload streams. Which regions are
        # recomputed or offloaded is decided by megatron.core.activation_policy.
        self.off_interface = _get_offloading_interface()
        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

        # Resolve the legacy `_forward_post_mlp` override once instead of walking the MRO on
        # every MLP bias-dropout-add. See _apply_mlp_bda_step for the deprecation contract.
        self._legacy_forward_post_mlp = None
        for klass in type(self).__mro__:
            if klass is TransformerLayer:
                break
            if "_forward_post_mlp" in vars(klass):
                self._legacy_forward_post_mlp = klass._forward_post_mlp
                break

        if (config.fp8 or config.fp4) and config.recompute_selected("layernorm"):
            # A recomputed norm's consumer saves the original norm output instead of its
            # quantized copy.
            if not is_identity(self.input_layernorm):
                self.self_attention.set_for_recompute_input_layernorm()
            if not is_identity(self.pre_mlp_layernorm):
                if self.is_moe_layer:
                    self.mlp.set_for_recompute_pre_mlp_layernorm()
                else:
                    from megatron.core.extensions.transformer_engine import set_save_original_input

                    set_save_original_input(self.mlp.linear_fc1)
        _set_offload_in_cuda_graph(self)

    def activation_strategies(self) -> list:
        """The norms and the MLP are child-module points; the residual adds are the method points
        _self_attn_bda and _mlp_bda.

        A half with a residual connection (wide residual, mHC) owns its carried residual: its norm
        offload group releases nothing, and when the layer stack replays the connected norms with
        their residual block (:meth:`_connected_norms_replayed`), the norm is not offloaded and
        its output is kept out of other offload groups.
        """
        config = self.config
        replayed = self._connected_norms_replayed()
        activation_strategies = []
        for norm, bda, returns_residual, connection, offload_name in (
            (
                self.input_layernorm,
                self._self_attn_bda,
                self._input_layernorm_returns_residual,
                self._get_residual_connection_for_self_attention(operation="read"),
                "attn_norm",
            ),
            (
                self.pre_mlp_layernorm,
                self._mlp_bda,
                self._pre_mlp_layernorm_returns_residual,
                self._get_residual_connection_for_mlp(operation="read"),
                "mlp_norm",
            ),
        ):
            if is_identity(norm):
                continue
            connected = connection is not None
            if config.recompute_selected("layernorm"):
                # The norm output is released when the branch output reaches the residual add.
                activation_strategies.append(
                    CheckpointWithoutOutputStrategy(
                        norm, release_on=Event(bda, "enter"), retain_input_tensors=returns_residual
                    )
                )
            if connected and replayed and config.fine_grained_activation_offloading:
                activation_strategies.append(ExcludeFromOffloadStrategy(norm))
            if config.offload_selected(offload_name):
                # Commits after the residual add, which consumed the residual (its input 1),
                # unless a residual connection owns the residual.
                activation_strategies.append(
                    OffloadStrategy(
                        norm,
                        offload_name,
                        commit_on=bda,
                        release=() if connected else (TensorRef.input(bda, 1),),
                        skip_when_replayed=connected,
                    )
                )
        if config.recompute_selected("mlp") and not self.is_moe_layer:
            activation_strategies.append(CheckpointStrategy(self.mlp))
        return activation_strategies

    def _connected_norms_replayed(self) -> bool:
        """Whether the layer stack replays the norms of connected halves with their residual
        block (see ``get_residual_branches``). No residual connections by default."""
        return False

    def init_backward_dw_wrapper(self):
        """Initialize ``self.backward_dw_wrapper`` for delayed-wgrad scheduling.

        The wrapper coordinates the per-layer wgrad callables (attention
        wgrad, optional shared-expert wgrad) with cuda-graph replay scope so
        captured components are not re-run eagerly. ``_BackwardDWWrapper``
        requires a ``TransformerLayer``; MambaLayer-derived modules implement
        ``backward_dw`` directly and skip this helper.
        """
        from megatron.core.models.common.utils import _BackwardDWWrapper

        config = getattr(self, 'config', None)
        assert config is not None, (
            "Module must be fully constructed (config set) before calling "
            "`init_backward_dw_wrapper`."
        )
        if config.cuda_graph_impl == "transformer_engine":
            from megatron.core.models.common.utils import _TECudaGraphBackwardDWWrapper

            self.backward_dw_wrapper = _TECudaGraphBackwardDWWrapper(self)
        else:
            self.backward_dw_wrapper = _BackwardDWWrapper(self)

    def get_inner_quantization_context(self) -> AbstractContextManager:
        """This layer's FP8/FP4 context under a per-layer recipe, entered around each call and
        by executors that run the layer's stages (none under delayed scaling, whose context the
        layer stack enters once)."""
        if self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed:
            from megatron.core.fp8_utils import get_fp8_context  # to avoid circular import

            return get_fp8_context(self.config, self.layer_number - 1)
        if self.config.fp4:
            from megatron.core.fp4_utils import get_fp4_context  # to avoid circular import

            return get_fp4_context(self.config, self.layer_number - 1)
        return nullcontext()

    def create_mcore_cudagraph_manager(self, config):
        """Register the transformer layer for cudagraphs."""

        assert self.config.cuda_graph_impl == "local"

        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        # If full scope (no specific sub-scope), cudagraph the entire layer.
        # Skip only when inference uses TransformerBlock-level graphs; otherwise the layer keeps
        # owning the empty-scope manager.
        if (
            not self.config.cuda_graph_modules
            and self.config.inference_cuda_graph_scope != InferenceCudaGraphScope.block
        ):
            self.cudagraph_manager = CudaGraphManager(config)
        elif (
            CudaGraphModule.attn in self.config.cuda_graph_modules
            and self.submodules_config.self_attention != IdentityOp
        ):
            self.cudagraph_manager = CudaGraphManager(config)
        elif (
            CudaGraphModule.mlp in self.config.cuda_graph_modules
            and self.submodules_config.mlp != IdentityOp
        ):
            # Cudagraphing MoE layers is handled by MoETransformerLayer.
            assert not self.is_moe_layer
            self.cudagraph_manager = CudaGraphManager(config)

    @staticmethod
    def _get_layer_offset(config: TransformerConfig):
        """
        Get the layer offset for the current pipeline stage.

        Deprecated: please use `get_transformer_layer_offset` instead.
        """

        warnings.warn(
            "TransformerLayer._get_layer_offset is deprecated."
            "Please use get_transformer_layer_offset instead."
        )
        return get_transformer_layer_offset(config)

    def _forward_self_attention_output_with_bias(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """Run input norm and self-attention, returning the raw output before BDA.

        The caller owns the residual add (see :meth:`_self_attn_bda`).

        Returns:
            ``(attention_output_with_bias, residual)``.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)

        input_layernorm_output, residual, attn_state = self._run_input_layernorm(hidden_states)
        if attn_state:
            raise RuntimeError(
                "Raw HybridStack attention execution requires an ordinary TransformerLayer "
                "without subclass-specific attention state."
            )

        using_fused_tp_inference_kernel = (
            InferenceMode.is_active() and self.config.inference_fuse_tp_communication
        )
        if using_fused_tp_inference_kernel:
            self._set_proj_residual(residual)

        nvtx_range_push(suffix="self_attention")
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        nvtx_range_pop(suffix="self_attention")
        return attention_output_with_bias, residual

    def supports_two_stage_attention(self) -> bool:
        """Return whether this is an attention-only layer that supports two-stage execution."""
        return (
            isinstance(self.self_attention, TwoStageAttentionLayer)
            and self.self_attention.supports_two_stage_attention()
            and isinstance(self.cross_attention, IdentityOp)
            and isinstance(self.mlp, IdentityOp)
        )

    def attention_bda_and_cross_attention(
        self,
        attention_output_with_bias,
        residual: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        attn_state=(),
    ):
        """Apply self-attention BDA and cross-attention."""
        hidden_states = self._apply_self_attn_bda_step(
            attention_output_with_bias, residual, attn_state
        )
        return self._run_cross_attention(hidden_states, context, context_mask, inference_context)

    def _forward_attention(
        self,
        # PyTorch's full-backward hooks only track positional inputs. Pass the
        # activation positionally so MFSDP can release weights promptly.
        # Supporting keyword inputs would require a custom autograd hook,
        # adding complexity to MFSDP.
        hidden_states: Tensor,
        /,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            rotary_pos_cos_sin (Optional[Tensor]): Combined rotary embedding cosine and sine.
            Currently used exclusively for inference with dynamic batching and flashinfer RoPE.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.
            input_ids (Tensor, optional): Token IDs retained in the shared layer-forward
                signature for the MLP phase. Self-attention does not consume them; hash-routed
                MoE layers use them later in ``_forward_mlp``.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """
        inference_context = deprecate_inference_params(inference_context, inference_params)
        input_layernorm_output, residual, attn_state = self._run_input_layernorm(hidden_states)

        using_fused_tp_inference_kernel = (
            InferenceMode.is_active() and self.config.inference_fuse_tp_communication
        )
        if using_fused_tp_inference_kernel:
            # Set the residual for fused reduce-scatter + add + layer-norm + all-gather
            # operation in attention's out_proj (linear_proj)
            self._set_proj_residual(residual)

        nvtx_range_push(suffix="self_attention")
        with _otel_managed_span('layer', 'megatron.layer.self_attention'):
            attention_output_with_bias = self.self_attention(
                input_layernorm_output,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
            )
        nvtx_range_pop(suffix="self_attention")

        return self.attention_bda_and_cross_attention(
            attention_output_with_bias,
            residual,
            context=context,
            context_mask=context_mask,
            inference_context=inference_context,
            attn_state=attn_state,
        )

    def forward_pre_attn_and_core_attn(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        packed_sequence_cp_metadata=None,
    ):
        """Run the training path through pre-attention and core attention."""
        assert self.supports_two_stage_attention()

        input_layernorm_output, residual, attn_state = self._run_input_layernorm(hidden_states)

        nvtx_range_push(suffix="self_attention")
        with _otel_managed_span('layer', 'megatron.layer.self_attention'):
            attention_intermediate = self.self_attention.forward_pre_attn_and_core_attn(
                input_layernorm_output,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                packed_sequence_cp_metadata=packed_sequence_cp_metadata,
            )
        nvtx_range_pop(suffix="self_attention")

        return attention_intermediate, residual, context, attn_state

    def forward_post_core_attn(
        self,
        attention_intermediate: Tensor,
        residual: Tensor,
        context: Optional[Tensor] = None,
        attn_state=(),
        context_mask: Optional[Tensor] = None,
    ):
        """Run the training path after core attention."""
        assert self.supports_two_stage_attention()

        attention_output_with_bias = self.self_attention.forward_post_core_attn(
            attention_intermediate
        )
        return self.attention_bda_and_cross_attention(
            attention_output_with_bias,
            residual,
            context=context,
            context_mask=context_mask,
            attn_state=attn_state,
        )

    def _validate_residual_branch_layout(self) -> tuple[bool, bool]:
        """Validate branch layouts for advanced residual-stream architectures.

        mHC and Wide Residual subclasses opt in after base construction; ordinary
        Transformer layers do not call this helper. Cross-attention is unsupported: a
        non-identity cross-attention module raises NotImplementedError, and a non-identity
        cross-attention norm or BDA raises ValueError. Inactive self-attention or MLP paths
        still execute at stream width, so a non-identity norm or BDA on them raises
        ValueError.

        Returns:
            Whether self-attention and the MLP are active.
        """
        if not isinstance(self.cross_attention, IdentityOp):
            raise NotImplementedError(
                f"{type(self).__name__} does not support cross-attention branches."
            )
        if not isinstance(self.pre_cross_attn_layernorm, IdentityOp) or not isinstance(
            self.cross_attn_bda, IdentityOp
        ):
            raise ValueError("Residual-stream layers require an identity cross-attention path.")

        has_attention = not isinstance(self.self_attention, IdentityOp)
        has_mlp = not isinstance(self.mlp, IdentityOp)
        for active, norm, bda, stage in (
            (has_attention, self.input_layernorm, self.self_attn_bda, "self-attention"),
            (has_mlp, self.pre_mlp_layernorm, self.mlp_bda, "MLP"),
        ):
            if not active and (not isinstance(norm, IdentityOp) or not isinstance(bda, IdentityOp)):
                raise ValueError(f"The inactive {stage} branch requires identity norm and BDA.")
        return has_attention, has_mlp

    def _get_residual_connection_for_self_attention(
        self, operation: ResidualConnectionOperation
    ) -> ResidualConnection | None:
        """Return the optional architecture-owned self-attention reader or writer."""

        match operation:
            case "read":
                return getattr(self, "residual_read_self_attn", None)
            case "write":
                return getattr(self, "residual_write_self_attn", None)
            case _:
                raise ValueError(f"Unsupported residual connection operation: {operation}.")

    def _get_residual_connection_for_mlp(
        self, operation: ResidualConnectionOperation
    ) -> ResidualConnection | None:
        """Return the optional architecture-owned MLP reader or writer."""

        match operation:
            case "read":
                return getattr(self, "residual_read_mlp", None)
            case "write":
                return getattr(self, "residual_write_mlp", None)
            case _:
                raise ValueError(f"Unsupported residual connection operation: {operation}.")

    def _is_last_residual_write(self, *, is_mlp: bool) -> bool:
        """Whether an active stage supplies the layer's last configured residual write."""
        return is_mlp or self._get_residual_connection_for_mlp(operation="write") is None

    def _read_residual_connection(
        self, connection: ResidualConnection, hidden_states: Tensor, *, is_mlp: bool
    ) -> tuple[Tensor, ResidualConnectionState]:
        """Invoke a stage's reader with the caller's dtype (replayed if residual replay hooks it)."""
        return apply_module(connection)(
            hidden_states,
            fp32_residual_connection=self.config.fp32_residual_connection,
            branch_input_dtype=self.config.params_dtype,
        )

    def _write_residual_connection(
        self,
        connection: ResidualConnection,
        branch_output: ResidualBranchOutput,
        state: ResidualConnectionState,
        *,
        is_mlp: bool,
    ) -> Tensor:
        """Invoke a stage's writer inside the shared schedule's gradient scope."""
        return apply_module(connection)(
            branch_output,
            state=state,
            dropout_probability=self.hidden_dropout,
            training=self.training,
        )

    def _split_norm_residual(
        self,
        norm_output,
        returns_residual: bool,
        hidden_states,
        connection,
        connection_state,
        norm_name: str,
    ):
        """Return ``(branch_input, residual)`` for a branch whose norm produced ``norm_output``.

        A configured residual connection owns the carried residual stream, so the norm must not
        return one. Otherwise a residual-returning (fused residual) norm supplies the residual,
        or it is the norm input.
        """
        if connection is not None:
            if isinstance(norm_output, tuple):
                raise ValueError(
                    f"{norm_name} cannot return a residual when a residual connection "
                    "owns the carried state."
                )
            return norm_output, connection.residual_stream(connection_state)
        if returns_residual:
            norm_output, residual = norm_output
        else:
            residual = hidden_states
        if self.config.fp32_residual_connection:
            residual = residual.float()
        return norm_output, residual

    def _run_input_layernorm(self, hidden_states):
        """Run the optional residual read and the input layernorm.

        Returns:
            Tuple ``(input_layernorm_output, residual, attn_state)`` where
            ``attn_state`` is an opaque payload subclasses can use to thread
            extra intermediates (e.g. mHC ``h_res``/``h_post``) through to
            ``_apply_self_attn_bda_step``. A configured residual connection uses
            it for the corresponding write state; otherwise base returns ``()``.
        """
        connection = self._get_residual_connection_for_self_attention(operation="read")
        connection_state = ()
        if connection is not None:
            hidden_states, connection_state = self._read_residual_connection(
                connection, hidden_states, is_mlp=False
            )
        output, residual = self._split_norm_residual(
            apply_module(self.input_layernorm)(hidden_states),
            self._input_layernorm_returns_residual,
            hidden_states,
            connection,
            connection_state,
            "input_layernorm",
        )
        return output, residual, connection_state

    def _apply_self_attn_bda_step(self, attention_output_with_bias, residual, attn_state=()):
        """Bias-dropout-add of the self-attention output."""
        nvtx_range_push(suffix="self_attn_bda")
        hidden_states = self._self_attn_bda(attention_output_with_bias, residual, attn_state)
        nvtx_range_pop(suffix="self_attn_bda")
        return hidden_states

    def _self_attn_bda(self, attention_output_with_bias, residual, attn_state=()):
        """Add the self-attention branch output to the residual stream.

        Subclasses may override this to consume custom intermediates threaded via
        ``attn_state``. Base uses that state for an optional residual connection and
        otherwise performs the ordinary bias-dropout-add.
        """
        if InferenceMode.is_active() and self.config.inference_fuse_tp_communication:
            # The residual add is already handled inside the self attention module.
            return attention_output_with_bias[0]
        connection = self._get_residual_connection_for_self_attention(operation="write")
        if connection is not None:
            if not attn_state:
                raise RuntimeError("Missing state for the self-attention residual connection.")
            # TODO: could we move `bias_dropout_add_exec_handler` itself
            # inside the module provided in the `bias_dropout_add_spec` module?
            with self.bias_dropout_add_exec_handler():
                return self._write_residual_connection(
                    connection, attention_output_with_bias, attn_state, is_mlp=False
                )
        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            return self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

    def _run_cross_attention(self, hidden_states, context, context_mask, inference_context):
        """Optional pre-cross-attn layernorm + cross-attention + bda block."""
        pre_cross_attn_layernorm_output = apply_module(self.pre_cross_attn_layernorm)(hidden_states)

        if self._pre_cross_attn_layernorm_returns_residual:
            pre_cross_attn_layernorm_output, residual = pre_cross_attn_layernorm_output
        else:
            residual = hidden_states

        if self.config.fp32_residual_connection:
            residual = residual.float()
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_context=inference_context,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        return hidden_states, context

    @copy_signature(_forward_attention)
    def forward(self, hidden_states: Tensor, /, *args, **kwargs):
        """
        Perform a forward pass through the transformer layer.

        This method calls the core computation of a transformer layer, including
        self-attention, cross-attention (if applicable), and feed-forward operations.
        """
        with _otel_managed_span(
            'layer', 'megatron.layer.forward', **{'megatron.layer_number': self.layer_number}
        ):
            hidden_states, context = self._forward_attention(hidden_states, *args, **kwargs)
            with _otel_managed_span('layer', 'megatron.layer.mlp'):
                mlp_kwargs = {
                    "padding_mask": kwargs.get("padding_mask", None),
                    "packed_seq_params": kwargs.get("packed_seq_params", None),
                }
                if kwargs.get("input_ids", None) is not None:
                    mlp_kwargs["input_ids"] = kwargs["input_ids"]
                output = self._forward_mlp(
                    hidden_states, kwargs.get("inference_context", None), **mlp_kwargs
                )
            return output, context

    def _maybe_unflatten_for_moe(self, hidden_states, padding_mask, input_ids, packed_seq_params):
        """Un-flatten packed sequences to restore the batch dimension for MoE.

        When inter-document masking flattens MBS > 1 into [mbs*S, 1, H], the MoE
        router sees bsz=1 and computes seq_aux_loss over the entire flattened
        sequence instead of per sample. Un-flattening to [S, mbs, H] before the
        MoE layer restores the correct per-sample structure.

        Returns:
            (hidden_states, padding_mask, input_ids, mbs) where mbs is None if no
            un-flattening was applied.
        """
        if (
            not self.is_moe_layer
            or packed_seq_params is None
            or getattr(packed_seq_params, 'tokens_per_sample', None) is None
        ):
            return hidden_states, padding_mask, input_ids, None

        tokens_per_sample = packed_seq_params.tokens_per_sample
        mbs = hidden_states.shape[0] // tokens_per_sample
        if mbs <= 1:
            return hidden_states, padding_mask, input_ids, None

        # The flattened tensor has all tokens from sample 0, then all tokens
        # from sample 1, etc. A plain reshape would keep that ordering, but we
        # need dim 0 to be the token position and dim 1 to be the sample index,
        # so view + transpose is required.
        hidden_states = hidden_states.view(mbs, tokens_per_sample, -1).transpose(0, 1).contiguous()
        if padding_mask is not None:
            padding_mask = padding_mask.view(mbs, tokens_per_sample)
        if input_ids is not None:
            input_ids = input_ids.view(mbs, tokens_per_sample)
        return hidden_states, padding_mask, input_ids, mbs

    def _maybe_reflatten_from_moe(self, output, packed_seq_params, mbs):
        """Re-flatten MoE output back to [mbs*S, 1, H] for the residual add."""
        if mbs is None:
            return output
        return output.transpose(0, 1).reshape(mbs * packed_seq_params.tokens_per_sample, 1, -1)

    def _pre_mlp_layernorm_and_residual(self, hidden_states):
        """Run the optional residual read and the pre-MLP layernorm.

        Returns:
            Tuple ``(pre_mlp_layernorm_output, residual, mlp_state)`` where
            ``mlp_state`` is an opaque payload subclasses can use to thread
            extra intermediates (e.g. mHC ``mlp_h_res`` / ``mlp_hc_h_post``)
            through to ``_apply_mlp_bda_step``. A configured residual connection
            uses it for the corresponding write state; otherwise base returns ``()``.
        """
        connection = self._get_residual_connection_for_mlp(operation="read")
        connection_state = ()
        if connection is not None:
            hidden_states, connection_state = self._read_residual_connection(
                connection, hidden_states, is_mlp=True
            )
        output, residual = self._split_norm_residual(
            apply_module(self.pre_mlp_layernorm)(hidden_states),
            self._pre_mlp_layernorm_returns_residual,
            hidden_states,
            connection,
            connection_state,
            "pre_mlp_layernorm",
        )
        return output, residual, connection_state

    def _forward_mlp_output_with_bias(
        self,
        hidden_states: Tensor,
        inference_context: BaseInferenceContext | None = None,
        padding_mask: Tensor | None = None,
        packed_seq_params=None,
        input_ids: Optional[Tensor] = None,
    ) -> tuple[tuple[Tensor, Tensor | None], Tensor]:
        """Run pre-MLP norm and MLP/MoE, returning the raw output before BDA.

        The caller owns the residual add (see :meth:`_mlp_bda`).
        """
        pre_mlp_layernorm_output, residual, mlp_state = self._pre_mlp_layernorm_and_residual(
            hidden_states
        )
        if mlp_state:
            raise RuntimeError(
                "Raw HybridStack MLP execution requires an ordinary TransformerLayer "
                "without subclass-specific MLP state."
            )

        pre_mlp_layernorm_output, padding_mask, input_ids, moe_unflatten_mbs = (
            self._maybe_unflatten_for_moe(
                pre_mlp_layernorm_output, padding_mask, input_ids, packed_seq_params
            )
        )

        mlp_output_with_bias = self._run_mlp(
            pre_mlp_layernorm_output,
            residual,
            padding_mask,
            inference_context,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
        )

        if moe_unflatten_mbs is not None:
            mlp_output, mlp_bias = mlp_output_with_bias
            mlp_output = self._maybe_reflatten_from_moe(
                mlp_output, packed_seq_params, moe_unflatten_mbs
            )
            mlp_output_with_bias = (mlp_output, mlp_bias)

        return mlp_output_with_bias, residual

    def _forward_mlp(
        self,
        hidden_states: Tensor,
        inference_context: BaseInferenceContext | None = None,
        padding_mask: Tensor | None = None,
        packed_seq_params=None,
        input_ids: Optional[Tensor] = None,
    ) -> Tensor | list[Tensor | None]:
        """
        Perform a forward pass through the feed-forward layer.

        Args:
            hidden_states (Tensor): Transformed hidden states before the MLP layernorm.
                Shape [seq_length, batch_size, hidden_size].
            inference_context: Inference context for optimizations.
            padding_mask (Tensor, optional): Padding mask for MoE routing.
                Shape [bsz, seq_length]. True = padding (exclude), False = valid (include).
                Only used for MoE layers to exclude padding tokens from aux loss computations.
                The MoELayer will internally transform this to [seq_length, bsz] format.
            packed_seq_params: Packed sequence parameters, used to detect flattened
                batches that need reshaping for MoE sequence load balancing.
            input_ids (Tensor, optional): Token IDs with shape [batch_size, seq_length].
                Required by hash-routed MoE layers.
        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """
        pre_mlp_layernorm_output, residual, mlp_state = self._pre_mlp_layernorm_and_residual(
            hidden_states
        )

        pre_mlp_layernorm_output, padding_mask, input_ids, moe_unflatten_mbs = (
            self._maybe_unflatten_for_moe(
                pre_mlp_layernorm_output, padding_mask, input_ids, packed_seq_params
            )
        )

        mlp_output_with_bias = self._run_mlp(
            pre_mlp_layernorm_output,
            residual,
            padding_mask,
            inference_context,
            input_ids=input_ids,
            packed_seq_params=packed_seq_params,
        )

        if moe_unflatten_mbs is not None:
            mlp_output, mlp_bias = mlp_output_with_bias
            mlp_output = self._maybe_reflatten_from_moe(
                mlp_output, packed_seq_params, moe_unflatten_mbs
            )
            mlp_output_with_bias = (mlp_output, mlp_bias)

        if self.config.cuda_graph_impl == "transformer_engine":
            cuda_graph_output = self._te_cuda_graph_mlp_capture_output(
                mlp_output_with_bias, residual
            )
            if cuda_graph_output is not None:
                return cuda_graph_output

        return self._apply_mlp_bda_step(mlp_output_with_bias, residual, mlp_state)

    def _run_mlp(
        self,
        pre_mlp_layernorm_output: Tensor,
        residual: Tensor,
        padding_mask: Tensor | None,
        inference_context: BaseInferenceContext | None,
        input_ids: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ):
        """Execute the MLP submodule with the appropriate variant.

        Picks between the chunked and direct-call paths; the direct call is the ``mlp`` region
        (checkpointed when ``"mlp"`` is selected for recompute). Shared by both
        :class:`TransformerLayer` and :class:`HyperConnectionTransformerLayer` so
        the MLP-call branching stays in one place.

        Returns:
            ``mlp_output_with_bias``: tuple of (mlp_output, mlp_bias).
        """
        nvtx_range_push(suffix="mlp")
        # Potentially chunk the MLP computation during prefill to minimize the peak activation size
        should_chunk_mlp_for_prefill = (
            self.config.mlp_chunks_for_prefill > 1
            and inference_context is not None
            and not inference_context.is_decode_only()
            and not isinstance(self.mlp, IdentityOp)
            and not self.config.transformer_impl == "inference_optimized"
        )
        should_chunk_mlp_for_training = (
            self.config.mlp_chunks_for_training > 1
            and inference_context is None
            and self.training
            and not isinstance(self.mlp, IdentityOp)
        )

        using_fused_tp_inference_kernel = (
            InferenceMode.is_active() and self.config.inference_fuse_tp_communication
        )
        moe_kwargs = {}
        if self.is_moe_layer and input_ids is not None:
            moe_kwargs["input_ids"] = input_ids
        if self.is_moe_layer:
            moe_kwargs["packed_seq_params"] = packed_seq_params

        if should_chunk_mlp_for_prefill or should_chunk_mlp_for_training:
            mlp_output_with_bias = self._run_mlp_in_chunks(
                pre_mlp_layernorm_output,
                padding_mask,
                input_ids,
                packed_seq_params,
                (
                    self.config.mlp_chunks_for_prefill
                    if should_chunk_mlp_for_prefill
                    else self.config.mlp_chunks_for_training
                ),
            )
        else:
            if using_fused_tp_inference_kernel:
                # Set the residual for fused reduce-scatter + add + layer-norm + all-gather
                # operation in MLP's fc2.
                self._set_fc2_residual(residual)
            mlp_output_with_bias = apply_module(self.mlp)(
                pre_mlp_layernorm_output, padding_mask=padding_mask, **moe_kwargs
            )

        nvtx_range_pop(suffix="mlp")
        return mlp_output_with_bias

    def _run_mlp_in_chunks(
        self,
        hidden_states: Tensor,
        padding_mask: Tensor | None,
        input_ids: Optional[Tensor],
        packed_seq_params: Optional[PackedSeqParams],
        num_chunks: int,
    ) -> tuple[Tensor, Tensor | None]:
        """Run the MLP over ``num_chunks`` sequence chunks of ``hidden_states`` ([s, b, h]).

        For an MoE layer, every call runs expert-parallel collectives, so every EP rank must make
        the same number of calls whatever its local sequence length. The input is therefore
        always split into exactly ``num_chunks`` non-empty chunks: a sequence shorter than that
        is padded with masked dummy tokens, whose outputs are dropped. With routing replay, each
        chunk replays its own tokens' indices (see ``ChunkedRouterReplay``).

        Returns:
            ``mlp_output_with_bias``: tuple of (mlp_output, mlp_bias).
        """
        seq_len, batch_size = hidden_states.shape[:2]
        router = getattr(self.mlp, "router", None)
        pad = 0
        if self.is_moe_layer:
            pad = max(0, num_chunks - seq_len)
        else:
            num_chunks = min(num_chunks, seq_len)
        if pad and getattr(router, "routing_type", None) == "quantile_balancing":
            # The dummy tokens need a padding mask, which quantile balancing does not support yet.
            raise NotImplementedError(
                f"MLP chunking (mlp_chunks_for_training / mlp_chunks_for_prefill = {num_chunks}) "
                f"with quantile-balancing routing needs a local sequence of at least {num_chunks} "
                f"tokens, got {seq_len}: shorter sequences are padded with masked dummy tokens, "
                "and quantile balancing does not support padding masks yet."
            )
        if pad:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 0, 0, pad))
            # Mask the dummy tokens even without a caller mask, so the router leaves them out
            # of its aux-loss, z-loss and expert-bias token counts.
            if padding_mask is None:
                padding_mask = torch.zeros(
                    (batch_size, seq_len), dtype=torch.bool, device=hidden_states.device
                )
            padding_mask = torch.nn.functional.pad(padding_mask, (0, pad), value=True)
            if input_ids is not None:
                input_ids = torch.nn.functional.pad(input_ids, (0, pad))

        # tensor_split, unlike chunk, always returns exactly num_chunks pieces.
        chunks = hidden_states.tensor_split(num_chunks, dim=0)
        padding_mask_chunks = (
            padding_mask.tensor_split(num_chunks, dim=1)
            if padding_mask is not None
            else (None,) * num_chunks
        )
        input_id_chunks = (
            input_ids.tensor_split(num_chunks, dim=1)
            if self.is_moe_layer and input_ids is not None
            else (None,) * num_chunks
        )
        replay = ChunkedRouterReplay(
            getattr(router, "router_replay", None),
            [chunk.shape[0] * batch_size for chunk in chunks],
            pad * batch_size,
            # MoELayer checkpoints each call under selective "moe" recompute.
            chunks_recompute=self.config.moe_calls_rerun_in_backward() and self.mlp.training,
        )

        # Compute outputs for each chunk, preserving sequence-aligned MoE metadata.
        outputs = []
        for i, (chunk, padding_mask_chunk, input_id_chunk) in enumerate(
            zip(chunks, padding_mask_chunks, input_id_chunks)
        ):
            chunk_moe_kwargs = {}
            if input_id_chunk is not None:
                chunk_moe_kwargs["input_ids"] = input_id_chunk
            if self.is_moe_layer:
                chunk_moe_kwargs["packed_seq_params"] = packed_seq_params
            replay_topk_idx = replay.chunk_indices(i)
            if replay_topk_idx is not None:
                chunk_moe_kwargs["replay_topk_idx"] = replay_topk_idx
            outputs.append(
                apply_module(self.mlp)(chunk, padding_mask=padding_mask_chunk, **chunk_moe_kwargs)
            )
            replay.after_chunk()
        replay.finish()

        # Aggregate chunk outputs, dropping the dummy tokens.
        mlp_output = torch.cat([out for out, _ in outputs], dim=0)[:seq_len]
        bias_chunks = [bias for _, bias in outputs if bias is not None]
        # elements in bias_chunks are the same for all chunks, so we can just use the first one
        bias_output = bias_chunks[0] if bias_chunks else None
        return mlp_output, bias_output

    def _apply_mlp_bda_step(
        self,
        mlp_output_with_bias: tuple[Tensor, Tensor | None],
        residual: Tensor,
        mlp_state: tuple = (),
    ) -> Tensor:
        """
        Bias-dropout-add of the MLP output (the ``mlp_bda`` region) + viewless-tensor wrap.

        Args:
            mlp_output_with_bias (Tensor): Output tensor of the MLP layer with bias.
            residual (Tensor): Residual tensor.
            mlp_state: Opaque payload from ``_pre_mlp_layernorm_and_residual``. Default ``()``.

        Returns:
            output (Tensor): Transformed hidden states of shape [s, b, h].
        """
        # Back-compat shim: prior to the MLP-hook refactor this method was named
        # `_forward_post_mlp` and took only (mlp_output_with_bias, residual). If a
        # subclass still overrides the legacy name, route through it and emit a
        # DeprecationWarning. `mlp_state` is dropped — the legacy contract didn't
        # have it. The override is resolved once in __init__ so this hot path stays
        # a single attribute check. To be removed in a future release.
        if self._legacy_forward_post_mlp is not None:
            warnings.warn(
                "TransformerLayer._forward_post_mlp has been renamed to "
                "_apply_mlp_bda_step and gained an `mlp_state` parameter. "
                "Override `_mlp_bda` instead; the legacy hook "
                "will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self._legacy_forward_post_mlp(self, mlp_output_with_bias, residual)

        nvtx_range_push(suffix="mlp_bda")
        hidden_states = self._mlp_bda(mlp_output_with_bias, residual, mlp_state)
        nvtx_range_pop(suffix="mlp_bda")

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        return make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

    def _mlp_bda(self, mlp_output_with_bias, residual, mlp_state=()):
        """Add the MLP branch output to the residual stream.

        Subclasses may override this to consume custom intermediates threaded via
        ``mlp_state``. Base uses that state for an optional residual connection and
        otherwise performs the ordinary bias-dropout-add.
        """
        if InferenceMode.is_active() and self.config.inference_fuse_tp_communication:
            # The residual add is already handled inside the MLP module.
            return mlp_output_with_bias[0]
        connection = self._get_residual_connection_for_mlp(operation="write")
        if connection is not None:
            if not mlp_state:
                raise RuntimeError("Missing state for the MLP residual connection.")
            # TODO: could we move `bias_dropout_add_exec_handler` itself
            # inside the module provided in the `bias_dropout_add_spec` module?
            with self.bias_dropout_add_exec_handler():
                return self._write_residual_connection(
                    connection, mlp_output_with_bias, mlp_state, is_mlp=True
                )
        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            return self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """
        Generate a sharded state dictionary for the transformer layer.

        Args:
            prefix (str, optional): Prefix to be added to all keys in the state dict.
            sharded_offsets (tuple, optional): Tuple of sharding offsets.
            metadata (Optional[dict], optional): Additional metadata for sharding.

        Returns:
            ShardedStateDict: A dictionary containing the sharded state of the transformer layer.
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        prefixed_map = {
            f'{prefix}{k}': f'{prefix}{v}'
            for k, v in self.submodules_config.sharded_state_dict_keys_map.items()
        }
        if prefixed_map:
            apply_prefix_mapping(sharded_state_dict, prefixed_map)
        return sharded_state_dict

    def configure_fused_tp_inference(
        self,
        skip_qkv_norm_and_all_gather: bool = False,
        fc2_next_layer_norm_weights: Optional[Tensor] = None,
    ):
        """
        Configure settings for fused TP communication in inference mode.

        Args:
            skip_qkv_norm (bool): Whether to skip norm and all-gather for linear_qkv.
            fc2_next_layer_norm_weights (Optional[Tensor]): Next layer's QKV norm weights
                for current layer's MLP FC2.
        """
        self.self_attention.linear_qkv.skip_norm_and_all_gather = skip_qkv_norm_and_all_gather

        # Use current layer's own MLP FC1 norm weights for attention's/mixer's out_proj
        mlp_fc1_weights = self.get_mlp_layer_norm_weights()
        self._set_proj_next_layer_norm_weights(mlp_fc1_weights)

        self.mlp.linear_fc1.skip_norm_and_all_gather = True
        # Use next layer's attention norm weights for current layer's MLP FC2
        self._set_fc2_next_layer_norm_weights(fc2_next_layer_norm_weights)

    def _set_proj_next_layer_norm_weights(self, weights: Tensor):
        """Set next layer norm weights for attention/mixer's linear_proj."""
        self.self_attention.linear_proj._set_next_layer_norm_weights(weights)

    def _set_fc2_next_layer_norm_weights(self, weights: Optional[Tensor]):
        """Set next layer norm weights for MLP FC2."""
        if weights is None:
            # Create dummy tensor for last layer (same shape as fc1 norm weights)
            weights = torch.empty_like(self.get_mlp_layer_norm_weights())
        self.mlp.linear_fc2._set_next_layer_norm_weights(weights)

    def _set_proj_residual(self, residual: Tensor):
        """Set residual for attention's/mixer's out_proj (linear_proj)."""
        self.self_attention.linear_proj._set_residual(residual)

    def _set_fc2_residual(self, residual: Tensor):
        """Set residual for MLP FC2."""
        self.mlp.linear_fc2._set_residual(residual)

    def get_mlp_layer_norm_weights(self) -> Tensor:
        """
        Get the MLP FC1 layer norm weights.

        Returns:
            Tensor: The layer norm weight data.
        """
        return self.mlp.linear_fc1.layer_norm_weight.data

    def get_qkv_layer_norm_weights(self) -> Tensor:
        """
        Get the QKV layer norm weights.

        Returns:
            Tensor: The layer norm weight data.
        """
        return self.self_attention.linear_qkv.layer_norm_weight.data

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        # Training and validation mode CUDA graphs.
        if (
            hasattr(self, 'cudagraph_manager')
            and kwargs.get('inference_context') is None
            and not torch.is_inference_mode_enabled()  # for inference eager dummy_forward
        ):
            return True
        # Inference mode. CUDA graphs are used in the decode phase only, when attn mask is None
        elif InferenceMode.is_active() and (
            hasattr(self, 'cudagraph_manager')
            # Owned by the model-level graph `_setup_mtp_cuda_graphs` installs; a nested
            # per-layer graph can be neither captured nor replayed ("Cannot prepare for
            # replay during capturing stage").
            and not self.is_mtp_layer
            and kwargs['attention_mask'] is None
            and (
                (kwargs.get('inference_context') is not None)
                or (kwargs.get('inference_params') is not None)
            )
            and not self.config.cuda_graph_modules  # empty-list = per-layer CUDA graphs
        ):
            if kwargs['inference_context'].is_static_batching():
                using_cuda_graph = kwargs['inference_context'].is_decode_only()
            else:
                # it can happen that non-decode steps have a token count greater than the max
                # supported cuda graph token count. In that case this flag will be set to
                # False by initialize_attention, and we should not use cuda graphs.
                using_cuda_graph = kwargs['inference_context'].using_cuda_graph_this_step()
            if using_cuda_graph:
                return True
        return False

    def get_layer_norm_weights(self):
        """
        Get the weights of all layernorms (attention and MLP) in the transformer layer.
        Returns:
            List[Tensor]: A list of layernorm weight tensors.
        """
        return


class TransformerLayer(TECudaGraphTransformerLayerMixin, _TransformerLayer):
    """A single transformer layer mapping [s, b, h] inputs to outputs of the same size.

    The TE mixin precedes the implementation so it can wrap attention and initialize
    graph state after layer construction. TE behavior is enabled only by config.
    """


class MoETransformerLayer(TransformerLayer):
    """
    A Transformer layer specialized for Mixture-of-Experts (MoE) architectures.

    Implements specific functionality to support CUDA graph capture for MoE layers.
    Due to the dynamic nature of MoE, capturing the entire layer in a single CUDA graph
    can be challenging. This class supports "partial" CUDA graphs by decomposing the
    MLP forward pass into router, expert-compute, and post-process stages.

    With partial CUDA graphs the router -> expert compute -> postprocess chain runs as
    :meth:`_forward_mlp_partial_cudagraphs`.
    """

    def __init__(self, *args, **kwargs):
        self.is_moe_layer = True
        self.use_partial_cudagraphs = False
        self._local_cudagraph_attr_names = None

        super().__init__(*args, **kwargs)

    def activation_strategies(self) -> list:
        """The TransformerLayer strategies, plus: with cuda_graph_impl='local', moe recompute
        checkpoints the partial-graph chain (MoELayer then skips its own "moe" checkpoint), in
        eval too."""
        activation_strategies = super().activation_strategies()
        if self.config.recompute_selected("moe") and self.config.cuda_graph_impl == "local":
            activation_strategies.append(CheckpointStrategy(self._forward_mlp_partial_cudagraphs))
        return activation_strategies

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Controls whether the full-layer cudagraph_manager captures the entire forward call
        as a single graph. Returns False to skip full-layer capture and route through _forward_mlp.

        MoE layers have two cudagraph modes:
        - Full-layer (use_partial_cudagraphs=False): the full-layer cudagraph_manager captures
          the forward pass as one graph. This is used during inference.
        - Partial (use_partial_cudagraphs=True): the full-layer manager is bypassed (returns
          False), and _forward_mlp routes through cudagraph_manager_router and
          cudagraph_manager_postprocess, which are monkey-patched onto _forward_mlp_router
          and _forward_mlp_postprocess by CudaGraphManager.__init__. The expert dispatch
          in between runs eagerly. This is used during training.
        """
        if self.use_partial_cudagraphs:
            return False
        if self.config.cuda_graph_impl != "local":
            return False
        return super()._should_call_local_cudagraph(*args, **kwargs)

    def transition_cudagraph_scope(self, mode):
        """Transition between full-layer and partial CUDA graph capture.

        Args:
            mode: 'full' for inference (full-layer capture) or 'partial' for training
            (router + postprocess captured, expert dispatch runs eagerly).
        """
        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        if mode == 'partial':
            self.use_partial_cudagraphs = True
            if not hasattr(self, 'cudagraph_manager_router'):
                self.cudagraph_manager_router = CudaGraphManager(
                    self.config, self, function_name="_forward_mlp_router"
                )
            if not hasattr(self, 'cudagraph_manager_postprocess'):
                self.cudagraph_manager_postprocess = CudaGraphManager(
                    self.config, self, function_name="_forward_mlp_postprocess"
                )
        elif mode == 'full':
            self.use_partial_cudagraphs = False
            self.mlp.fwd_execution_map = ["route", "expert_compute", "postprocess"]
            assert hasattr(self, 'cudagraph_manager'), (
                "MoETransformerLayer missing full cudagraph_manager; "
                "expected it to be created at __init__ with scope = [] "
            )
        else:
            raise ValueError(f"Unknown MoE cudagraph mode: {mode}, expected 'full' or 'partial'")

    def create_mcore_cudagraph_manager(self, config):
        """
        Initializes the CUDA graph manager(s) for the MoE layer.

        Unlike the standard layer which typically uses a single manager, this method
        can configure multiple graph managers if partial CUDA graphs are enabled via
        `cuda_graph_modules`. This allows capturing the static parts of the MoE pass
        while leaving the expert computation to execute eagerly.
        """

        assert self.config.cuda_graph_impl == "local"

        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        if (
            not self.config.cuda_graph_modules
            and self.config.inference_cuda_graph_scope != InferenceCudaGraphScope.block
        ) or CudaGraphModule.moe in self.config.cuda_graph_modules:
            self.cudagraph_manager = CudaGraphManager(config)
        elif (
            CudaGraphModule.moe_router in self.config.cuda_graph_modules
            or CudaGraphModule.moe_preprocess in self.config.cuda_graph_modules
        ):
            self.transition_cudagraph_scope('partial')

    def _resolve_token_dispatcher_attr(self, attr_name: str) -> tuple[Any, str]:
        parent_attr_name, _, leaf_attr_name = attr_name.rpartition('.')
        obj = self.mlp.token_dispatcher
        for parent_name in parent_attr_name.split('.') if parent_attr_name else ():
            obj = getattr(obj, parent_name)
        return obj, leaf_attr_name or attr_name

    def _restore_token_dispatcher_attrs(self, attr_outputs):
        assert len(attr_outputs) == len(self._local_cudagraph_attr_names)
        for attr_name, attr in zip(self._local_cudagraph_attr_names, attr_outputs):
            obj, name = self._resolve_token_dispatcher_attr(attr_name)
            setattr(obj, name, attr)

    def _get_token_dispatcher_attrs(self):
        attr_names = []
        token_dispatcher_attr_outputs = []
        for attr_name in self.mlp.token_dispatcher.cudagraph_attrs:
            obj, name = self._resolve_token_dispatcher_attr(attr_name)
            attr = getattr(obj, name)
            if torch.is_tensor(attr):
                attr_names.append(attr_name)
                token_dispatcher_attr_outputs.append(attr)

        return tuple(attr_names), token_dispatcher_attr_outputs

    def _synchronize_router_host_outputs(self, attr_outputs):
        """Wait for partial-router graph outputs only when they reside on the host."""
        if not any(attr.device.type == "cpu" for attr in attr_outputs):
            return

        if not hasattr(self, '_router_dtoh_event'):
            self._router_dtoh_event = torch.cuda.Event()
        self._router_dtoh_event.record()
        self._router_dtoh_event.synchronize()

    def _forward_mlp_router(self, hidden_states, padding_mask=None, input_ids=None):
        """
        Executes the router phase of the MoE block.

        This includes the pre-MLP layernorm and the routing logic.
        This method is isolated so it can be captured by `cudagraph_manager_router`.
        """

        self.mlp.fwd_execution_map = "route"
        pre_mlp_layernorm_output, residual, _ = self._pre_mlp_layernorm_and_residual(hidden_states)

        hidden_states, probs, shared_expert_output = apply_module(self.mlp)(
            pre_mlp_layernorm_output,
            intermediate_tensors=(),
            padding_mask=padding_mask,
            input_ids=input_ids,
        )

        if self.use_partial_cudagraphs:
            attr_names, token_dispatcher_attr_outputs = self._get_token_dispatcher_attrs()
            if self._local_cudagraph_attr_names is None:
                self._local_cudagraph_attr_names = attr_names
            else:
                assert attr_names == self._local_cudagraph_attr_names
        else:
            # For eager mode, no need to pass the token_dispatcher attributes
            token_dispatcher_attr_outputs = []

        return (
            residual,
            hidden_states,
            probs,
            shared_expert_output,
            *token_dispatcher_attr_outputs,
        )

    def _forward_mlp_expert_compute(self, hidden_states, probs, token_dispatcher_attr_outputs):
        """
        Executes the actual computation of the experts.

        This phase takes the routing information and inputs, dispatches them to the
        appropriate experts, and computes the results. In partial graph modes, this
        step runs eagerly between the router and postprocess graph replays.
        """

        if self.use_partial_cudagraphs:
            # Restore the token dispatcher attrs returned on the router graph's output surface.
            self._restore_token_dispatcher_attrs(token_dispatcher_attr_outputs)

        self.mlp.fwd_execution_map = "expert_compute"
        return apply_module(self.mlp)(None, intermediate_tensors=(hidden_states, probs))

    def _forward_mlp_postprocess(self, residual, output, shared_expert_output, mlp_bias):
        """
        Executes the post-processing phase of the MoE block.

        Handles combining the expert outputs, applying biases, re-registering
        activation recomputation hooks if necessary, and performing the final
        Bias-Dropout-Add. This method is isolated so it can be captured by cudagraphs.

        """

        self.mlp.fwd_execution_map = "postprocess"
        output = apply_module(self.mlp)(None, intermediate_tensors=(output, shared_expert_output))
        return self._apply_mlp_bda_step((output, mlp_bias), residual)

    def _forward_mlp_partial_cudagraphs(
        self, hidden_states, inference_context=None, padding_mask=None, input_ids=None
    ):
        """The router -> expert compute -> postprocess chain run with partial CUDA graphs."""
        router_outputs = self._forward_mlp_router(
            hidden_states, padding_mask=padding_mask, input_ids=input_ids
        )
        residual, hidden_states, probs, shared_expert_output, *token_dispatcher_attr_outputs = (
            router_outputs
        )

        # CUDA outputs remain ordered by the graph-completion event. Only host outputs need
        # a CPU-blocking wait before the eager dispatcher can consume them.
        self._synchronize_router_host_outputs(token_dispatcher_attr_outputs)

        expert_output, mlp_bias = self._forward_mlp_expert_compute(
            hidden_states, probs, token_dispatcher_attr_outputs
        )
        return self._forward_mlp_postprocess(
            residual, expert_output, shared_expert_output, mlp_bias
        )

    def _forward_mlp(
        self,
        hidden_states,
        inference_context=None,
        padding_mask=None,
        packed_seq_params=None,
        input_ids=None,
    ):
        """
        Orchestrates the MLP forward pass, handling partial CUDA graph execution logic.

        If `use_partial_cudagraphs` is True, this method stitches together the
        router, expert_compute, and postprocess calls.
        """

        if inference_context is not None:
            assert not self.use_partial_cudagraphs, (
                "Partial cudagraphs for MoEs were detected during inference!"
                "Please do not use --cuda-graph-modules moe_router moe_preprocess "
                "alongside inference."
            )

        if self.use_partial_cudagraphs:
            hidden_states, padding_mask, input_ids, moe_unflatten_mbs = (
                self._maybe_unflatten_for_moe(
                    hidden_states, padding_mask, input_ids, packed_seq_params
                )
            )

            result = self._forward_mlp_partial_cudagraphs(
                hidden_states, padding_mask=padding_mask, input_ids=input_ids
            )

            result = self._maybe_reflatten_from_moe(result, packed_seq_params, moe_unflatten_mbs)

            return result
        else:
            return super()._forward_mlp(
                hidden_states,
                padding_mask=padding_mask,
                packed_seq_params=packed_seq_params,
                input_ids=input_ids,
            )


def __getattr__(name):
    """Resolve the legacy mHC layer import to its canonical implementation."""
    if name == "HyperConnectionTransformerLayer":
        from megatron.core.transformer.hyper_connection_transformer_layer import (
            HyperConnectionTransformerLayer,
        )

        return HyperConnectionTransformerLayer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
