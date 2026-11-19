# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Every module with activation strategies builds and installs them in its own ``__init__``.

Installing replaces the strategies installed before, so a subclass whose strategies differ from
its base class's (mHC and wide-residual layers, MLA variants, MoE layers) installs again at the
end of its own ``__init__``; otherwise its instances silently keep the base class's strategies.
"""

import importlib
import inspect

import pytest

# (module, class) of every class that installs activation strategies.
_OWNERS = [
    ("megatron.core.transformer.transformer_layer", "_TransformerLayer"),
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


@pytest.mark.parametrize("module,name", _OWNERS, ids=[name for _, name in _OWNERS])
def test_owner_installs_in_its_own_init(module, name):
    cls = _cls(module, name)
    assert "__init__" in vars(cls), f"{name} must define __init__ to install its strategies"
    source = inspect.getsource(vars(cls)["__init__"])
    assert "install_activation_strategies" in source, f"{name}.__init__ does not install"
