# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Every module with activation strategies builds them in ``activation_strategies``.

Construction installs ``activation_strategies()`` once the module is fully built, so a subclass
whose strategies differ from its base class's (mHC and wide-residual layers, MLA variants, MoE
layers) overrides the method instead of installing again.
"""

import importlib
import inspect

import pytest

from megatron.core.transformer.module import MegatronModule

# (module, class) of every class that installs activation strategies.
_OWNERS = [
    ("megatron.core.transformer.transformer_layer", "TransformerLayer"),
    ("megatron.core.transformer.transformer_layer", "MoETransformerLayer"),
    ("megatron.core.transformer.wide_residual_transformer_layer", "WideResidualTransformerLayer"),
    (
        "megatron.core.transformer.hyper_connection_transformer_layer",
        "HyperConnectionTransformerLayer",
    ),
    ("megatron.core.transformer.hyper_connection", "HyperConnectionModule"),
    ("megatron.core.transformer.attention", "Attention"),
    ("megatron.core.transformer.multi_latent_attention", "MultiLatentAttention"),
    (
        "megatron.core.transformer.experimental_attention_variant.absorbed_mla",
        "AbsorbedMLASelfAttention",
    ),
    (
        "megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention",
        "DSv4HybridAttention",
    ),
    ("megatron.core.transformer.moe.moe_layer", "MoELayer"),
    ("megatron.core.transformer.moe.experts", "TEGroupedMLP"),
    ("megatron.core.ssm.gated_delta_net.common", "_GDNBase"),
    ("megatron.core.ssm.gated_delta_product", "GatedDeltaProductMixer"),
    ("megatron.core.models.hybrid.shortcut_block", "ShortcutMoEBlock"),
    ("megatron.core.transformer.multi_token_prediction", "MultiTokenPredictionLayer"),
    ("megatron.core.transformer.transformer_block", "TransformerBlock"),
    ("megatron.core.models.hybrid.hybrid_block", "HybridStack"),
    ("megatron.core.models.audio.nemo_transformer_encoder", "TransformerEncoder"),
    ("megatron.core.models.audio.nemo_rope_transformer_encoder", "RopeTransformerEncoder"),
]


def _cls(module, name):
    return getattr(importlib.import_module(module), name)


# Owners that are not Megatron modules install explicitly at the end of ``__init__``.
_EXPLICIT_INSTALL = {"TransformerEncoder", "RopeTransformerEncoder"}


@pytest.mark.parametrize("module,name", _OWNERS, ids=[name for _, name in _OWNERS])
def test_owner_defines_its_strategies(module, name):
    cls = _cls(module, name)
    if name in _EXPLICIT_INSTALL:
        source = inspect.getsource(vars(cls)["__init__"])
        assert "install_activation_strategies" in source, f"{name}.__init__ does not install"
        return
    method = cls.activation_strategies
    assert method is not MegatronModule.activation_strategies, f"{name} defines no strategies"
    assert "install_activation_strategies" not in inspect.getsource(
        vars(cls).get("__init__", lambda: None)
    ), f"{name}.__init__ installs strategies itself"
