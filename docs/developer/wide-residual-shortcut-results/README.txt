Recorded GB300 measurements, 2026-10-15

cleanup/: paired pre/post cleanup results for learned and disabled retention.
unfused_comparison/: true retention=None on both fused and unfused paths.

The REPORT.txt files document protocols, scope and limitations. Raw CUDA event
samples are included. Historical reports reference the co-located standalone
shortcut_block_bench checkout, where complete scripts, frozen controls, assembly
and sanitizer logs remain archived under results/. Large profiler binaries and
source snapshots are not duplicated here.

These kernel timings exclude GEMMs and are not end-to-end training measurements.
The previously unmet Fusion B >80% DRAM-utilization target remains unresolved.
