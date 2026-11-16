__all__ = ["Detector"]


def __getattr__(name):
    # lazy: `import detection.paths` must not pull in torch + pcdet
    if name == "Detector":
        from detection.detector import Detector
        return Detector
    raise AttributeError(f"module 'detection' has no attribute {name!r}")
