from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Frame:
    """Reprezentuje pojedynczą klatkę w sekwencji"""
    sequence_id: str
    frame_id: int
    points: np.ndarray
    calib: dict = field(default_factory=dict)
    labels_gt: Optional[list] = None

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] < 3:
            raise ValueError(
                f"Shape of points array is expected to be (N, 3) or (N, 4), received {self.points.shape}"
            )

    @property
    def num_points(self) -> int:
        """Returns number of points in the frame"""
        return self.points.shape[0]

    @property
    def xyz(self) -> np.ndarray:
        """Returns coordinates of the point without intensity"""
        return self.points[:, :3]

    def __repr__(self) -> str:
        return (
            f"Frame(sequence={self.sequence_id!r}, frame_id={self.frame_id}, num_points={self.num_points})"
        )