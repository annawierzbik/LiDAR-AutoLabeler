# 0007 `gpu` test marker

Status: accepted (phase 1)

## Decision
Tests that need CUDA, OpenPCDet, the model weights or the KITTI data carry `@pytest.mark.gpu`
(registered in `pytest.ini`) and are deselected by default (`addopts = -m "not gpu"`). Run them
with `pytest -m gpu`. Their heavy imports (torch, pcdet) happen inside fixtures, so collecting the
module costs nothing. All other tests use fakes (fake builder, fake detector, `LoaderPointSource`
over fake frames) and never call the detector.

Affected: `tests/test_detection/test_detector.py` (was a module-level CUDA skip) and
`test_fov.py::test_camera_mask_matches_openpcdet_get_fov_flag`.
