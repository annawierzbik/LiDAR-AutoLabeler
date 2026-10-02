from pathlib import Path
from typing import Iterator
from .frame import Frame
from .loader import load_point_cloud, parse_calib_file, parse_tracking_labels

class SequenceLoader:
    def __init__(self, data_root, split, sequence):
        self.data_root = Path(data_root)
        self.split = split
        self.sequence = sequence

        self.velodyne_dir = self.data_root / split / "velodyne" / sequence
        self.calib_path = self.data_root / split / "calib" / f"{sequence}.txt"
        self.label_path = self.data_root / split / "label_02" / f"{sequence}.txt"

        print("velodyne_dir:", self.velodyne_dir)
        print("calib_path:", self.calib_path)
        print("label_path:", self.label_path)

        if not self.velodyne_dir.exists():
            raise FileNotFoundError(
                f"Directory with point clouds not found: {self.velodyne_dir}"
            )

        self._frame_files = sorted(self.velodyne_dir.glob("*.bin"))
        if not self._frame_files:
            raise FileNotFoundError(f"Directory {self.velodyne_dir} does not contain .bin files")

        self._calib = parse_calib_file(self.calib_path) if self.calib_path.exists() else {}
        self._labels_by_frame = (parse_tracking_labels(self.label_path) if self.label_path.exists() else {})

    def __len__(self):
        return len(self._frame_files)

    def _load_frame(self, index):
        bin_path = self._frame_files[index]
        frame_id = int(bin_path.stem)

        points = load_point_cloud(bin_path)
        labels_gt = self._labels_by_frame.get(frame_id, [])

        return Frame(
            sequence_id=self.sequence,
            frame_id=frame_id,
            points=points,
            calib=self._calib,
            labels_gt=labels_gt,
        )

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Index {index} out of bounds [0, {len(self)})")
        return self._load_frame(index)

    def __iter__(self) -> Iterator[Frame]:
        for i in range(len(self)):
            yield self._load_frame(i)