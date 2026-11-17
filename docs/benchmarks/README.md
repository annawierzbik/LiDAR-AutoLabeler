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
- `gpu_smoke_stack_memory.csv`: total GPU memory in use (MiB) during
  `python scripts/smoke_stack.py --gpu` on 2026-11-17 (the containerised stack with the real
  worker-gpu image, 6 frames of KITTI tracking 0000, the whole browser scenario through nginx),
  sampled the same way (`nvidia-smi ... -lms 250`). Script start 21:43:25, stack healthy 21:44:02,
  tests 21:44:03-21:44:42, teardown done 21:44:46; the whole run took 81 s (images already built).
  Before: 2299 MiB; peak 2870 MiB at 21:44:18 (during the scenario), about 571 MiB above it; after
  the teardown: 2149-2150 MiB (other applications changed their use meanwhile, the last sample
  is 2336 MiB). Limits as above: whole GPU, Windows/WDDM, 250 ms sampling, one run. A unit-test
  loop ran on the CPU at the same time.
