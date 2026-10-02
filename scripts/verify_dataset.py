from pathlib import Path
import sys

# Dodaj główny katalog projektu do ścieżki, aby móc importować moduły ze src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.loader import load_point_cloud, parse_calib_file, parse_tracking_labels


def verify_kitti_sequence(data_root: str, split: str, sequence: str) -> bool:
    """Verifies the existence and structural integrity of a KITTI sequence."""
    data_path = Path(data_root)
    velodyne_dir = data_path / split / "velodyne" / sequence
    calib_file = data_path / split / "calib" / f"{sequence}.txt"
    label_file = data_path / split / "label_02" / f"{sequence}.txt"

    print(f"\n[Verifying Sequence: {sequence} (split: {split})]...")

    # 1. Check directories and files existence
    if not velodyne_dir.exists():
        print(f"  [ERROR] Velodyne directory missing: {velodyne_dir}")
        return False
    
    bin_files = sorted(list(velodyne_dir.glob("*.bin")))
    if not bin_files:
        print(f"  [ERROR] No .bin point cloud files found in {velodyne_dir}")
        return False
    print(f"  [OK] Found {len(bin_files)} point cloud frames.")

    if not calib_file.exists():
        print(f"  [WARNING] Calibration file missing: {calib_file}")
    else:
        try:
            calib = parse_calib_file(calib_file)
            print(f"  [OK] Calibration parsed successfully ({len(calib)} keys found).")
        except Exception as e:
            print(f"  [ERROR] Failed to parse calibration file: {e}")
            return False

    if not label_file.exists():
        print(f"  [WARNING] Label file missing: {label_file}")
    else:
        try:
            labels = parse_tracking_labels(label_file)
            print(f"  [OK] Labels parsed successfully ({len(labels)} frames annotated).")
        except Exception as e:
            print(f"  [ERROR] Failed to parse label file: {e}")
            return False

    # 2. Spot check first and last point cloud file integrity
    for sample_file in [bin_files[0], bin_files[-1]]:
        try:
            points = load_point_cloud(sample_file)
            if points.shape[1] < 3:
                print(f"  [ERROR] Invalid point shape in {sample_file.name}: {points.shape}")
                return False
        except Exception as e:
            print(f"  [ERROR] Failed to load point cloud {sample_file.name}: {e}")
            return False

    print(f"  [SUCCESS] Sequence {sequence} is fully valid!")
    return True


if __name__ == "__main__":
    DATA_ROOT = "data/raw"
    SPLIT = "training"
    
    # Podaj sekwencje, które faktycznie pobrałeś (np. ["0000", "0001"])
    SEQUENCES_TO_CHECK = ["0000", "0001"]

    success = True
    for seq in SEQUENCES_TO_CHECK:
        seq_dir = Path(DATA_ROOT) / SPLIT / "velodyne" / seq
        if seq_dir.exists():
            if not verify_kitti_sequence(DATA_ROOT, SPLIT, seq):
                success = False
        else:
            print(f"\n[INFO] Sequence {seq} folder not found in {seq_dir}, skipping...")

    if success:
        print("\nAll checked sequences passed verification successfully!")
    else:
        print("\nVerification finished with errors. Please check logs above.")