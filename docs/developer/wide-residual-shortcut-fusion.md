# Wide-residual shortcut MoE kernel fusion

Enable fused residual write/read/RMSNorm boundaries in the eager hybrid shortcut
schedule with:

```bash
--moe-shortcut-connection \
--wide-residual 3 \
--wide-residual-shortcut-moe-kernel-fusion \
--bf16 --fp32-residual-connection \
--normalization RMSNorm --layernorm-epsilon 1e-5 \
--disable-bias-linear --hidden-dropout 0 \
--moe-shared-expert-intermediate-size 1024
```

The flag defaults to off. Canonical sharded checkpoint names are preserved.
Built-in training specs select static fusion variants with an external input
RMSNorm; this changes its
Python module ownership and raw `state_dict` key. Custom specs must expose an
external input norm.
`TransformerConfig.wide_residual_shortcut_moe_kernel_fusion=True` is the equivalent
Core API setting.

The supported architecture pairs a compute-only `WideResidualMambaLayer`
(including GDP) or `WideResidualTransformerLayer` (including GDN/GDN2) with
shortcut MoE and a shared expert. Ordinary RMSNorm weights and branch updates
are BF16; carried residuals and internal reads are FP32.
`--fp32-residual-connection` is required. The more accurate FP32 reads can change
values from the ordinary schedule's BF16 read boundaries; byte equality with the
unfused model is not the numerical contract.

Construction checks the architecture, BF16 parameters, RMSNorm epsilon 1e-5,
ordinary gamma, and bias/dropout-free boundaries. Learned retention is optional;
when enabled its maximum forget rate must be 0.1. The current schedule rejects
shortcut post-norm, shared-expert overlap, context parallelism, full-layer
recomputation and offloading outside the expert leaf. Expert activation offloading
with `--offload-modules fused_group_mlp` remains owned by the existing MoE
module and is supported. Latent MoE (including its optional norm before the
up-projection), shortcut communication overlap, and selective recomputation are
supported. Latent projections remain in MoE preprocess/postprocess. MTP retains
its ordinary-width shortcut path. Inference contexts and packed/padded
input layouts are rejected at the call boundary.

Runtime shape validation runs before fused work:

- `H` is the ordinary hidden size: 32–10240, divisible by 32.
- `T` is the product of the local leading dimensions, including microbatch size:
  a positive multiple of 1024. The maximum is 65536 for `H <= 4096`, otherwise
  16384. These are local shapes after any sequence partitioning.
- Residuals are contiguous FP32 tensors of shape `[..., S*H]`. The existing
  wide-residual model configuration permits `S=2..6` for this flag. The direct
  kernel API additionally supports `S=1`.

A fuses the compute-layer write with the shared expert's read and RMSNorm. B
fuses the MoE write with the next adjacent pair's compute and routed-expert
reads/RMSNorms. Handoffs are call-local; the final pair returns a tensor and uses
its ordinary terminal write. Projection, router, expert, collective and parameter
modules retain their existing ownership.

By default the fused stack keeps each layer's residual output for backward.
Set `--residual-stream-recompute-num-layers N` to keep periodic residual anchors
and pass inverse outputs backward between writes. `N=16` keeps anchors at most
16 physical layers apart, plus the stack input and final output. Compute/MoE
pairs remain intact, so a block can end early rather than split a pair. `None`
(omitted) or `N=1` keeps the default behavior. No other recompute flag is needed;
selecting `residual_stream` alone does not enable this inverse chain. Unfused
checkpoint defaults are unchanged.

A/B forward runs once. Each kernel's existing backward restores the preceding
write's carried residual before returning gradients to its readers and branch
consumers. Ordinary streamwise writes use an inverse-only bridge, preserving
their existing forward and gradient kernels. A routed reader that becomes ready
before its producer's full backward requests inverse-only restoration from a
ready successor; this also handles pruned auxiliary-loss branches. There is no residual forward replay
or nested checkpoint backward. Intermediate residual storage is released after
its forward consumers finish and restored one write at a time during backward;
periodic anchors retain the original FP32 values and bound inverse rounding
error. This is a first-order linear write chain of contiguous full-storage FP32
residuals. Branch tensors and expert/mixer activation ownership are unchanged.
FP32 inversion is not bit exact because forward rounding discards information;
BF16 gradients can differ from the save-every-layer policy. Selective GDP and
expert recomputation retain their existing independent behavior.
Inverse intervals require the native ordinary streamwise writer: H=32
is rejected at construction, and shapes exceeding its controller-reduction
capacity reject at runtime. Use interval 1 for those shapes; no forward replay
is substituted.

Backward reconstructs from the FP32 carried residual, saved branch update,
coefficients and RMS statistics; it does not retain the original residual or
RMSNorm input. The independent three-stream cooperative kernel and its dispatch
are preserved. Other counts use their qualified cooperative schedules or standard
backward. New cooperative specializations require at least two resident CTAs per
SM; the measured S=6 production configurations use standard backward. There is
no runtime tuning or profiling.

The implementation lives entirely under
`megatron/core/fusions/wide_residual_shortcut/` and
`megatron/core/models/hybrid/layers/shortcut_fusion.py`; it imports no benchmark
checkout. Imports of Triton/Gluon are deferred until the fusion is used.

## Validation

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/fusions/test_wide_residual_shortcut.py \
  tests/unit_tests/models/test_wide_residual_shortcut_fusion.py \
  tests/unit_tests/determinism/kernels/test_wide_residual_shortcut.py \
  tests/unit_tests/models/test_shortcut_block.py \
  tests/unit_tests/transformer/test_wide_residual_connection.py \
  tests/unit_tests/determinism/kernels/test_manifest.py
```

Tests cover an independent FP64 formula for A/B forward and complete backward,
all six kernel stream counts, non-128-column widths, token multiples, controller
padding, unused outputs, frozen parameters, independent live handoffs, unchanged
state keys, CLI/configuration rejection and bit-exact output/gradient replay.
Model tests run the actual `HybridStack` loop, built-in compute specs, and
TE-backed routed/shared experts with all-gather and all-to-all dispatchers,
including optimizer steps and two-rank expert parallelism. A production Super
shape exercises cooperative backward with both dispatchers and real MoE
communication. End-to-end throughput and training convergence remain separate
qualification work.

Two independent GB300 processes compared the port with the same-format
three-stream incumbent using three rotating pools, 60 calls per graph, five
replays and nine randomized rounds, with CPU isolation and native timing outside
profilers. Forward and backward each passed the strict 1% regression limit at
Nano, Super and Ultra; the worst measured regression was 0.297%. All compared
forward and main-backward SASS was identical. The port also passed unfiltered
memory, race and synchronization checks on complete public-autograd calls.
