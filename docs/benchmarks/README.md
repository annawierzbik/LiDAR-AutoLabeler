# Benchmarks

- `session_store_phase2_before.csv`, `session_store_phase2b_after.csv`: session store timings, see
  ADR 0011.
- `gpu_worker_memory.csv`: total GPU memory in use (MiB) while
  `pytest -m "gpu and worker" tests/test_worker/test_gpu_worker.py` ran at commit d59b1d6
  (2026-11-16, NVIDIA GeForce RTX 4070, 12282 MiB, driver 610.88; 20 frames of KITTI tracking 0000,
  9.0 s including the model load). Sampled with
  `nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 250`, started
  before the test and stopped after it. Before: about 1715 MiB (desktop applications); peak
  2245 MiB, i.e. about 530 MiB above it; after: 1715 MiB. Limits: on Windows (WDDM) `nvidia-smi`
  reports no per-process memory, so the figure is the whole GPU minus the baseline, and other
  applications may have changed their use meanwhile; a spike shorter than 250 ms can be missed;
  one run, one sequence, 20 frames.
