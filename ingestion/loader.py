from pathlib import Path

import numpy as np


def load_point_cloud(bin_path):
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"File {bin_path} not found")
    
    points = np.fromfile(str(bin_path), dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(
            f"Invalid size of data in {bin_path}: {points.size} is not divisible by 4 (x, y, z, intensity)"
        )
    return points.reshape(-1, 4)

def parse_calib_file(calib_path):
    """Parsuje plik kalibracji KITTI do słownika macierzy numpy.

    Plik tekstowy, linia po linii w formacie "KLUCZ: wartość1 wartość2 ...".
    Liczba wartości w linii decyduje o kształcie macierzy po przeliczeniu:

        12 liczb -> macierz 3x4 (rotacja + translacja połączone w jedną macierz)
                    dotyczy kluczy: P0, P1, P2, P3 (macierze projekcji kamer),
                    Tr_velo_to_cam / Tr_velo_cam (transformacja LiDAR -> kamera)
         9 liczb -> macierz 3x3 (czysta rotacja)
                    dotyczy klucza: R_rect / R0_rect (macierz rektyfikacji)
        inna liczba -> zostaje jako płaska tablica 1D (nieznany/nieobsłużony format)

    Najważniejsze klucze dla LiDAR-owej detekcji 3D:
        - Tr_velo_to_cam -> transformacja punktu z układu LiDAR do układu kamery
        - R_rect           -> rektyfikacja (korekta zniekształceń optycznych kamery)
        - P2                -> projekcja z układu kamery rektyfikowanej na piksele
                              obrazu (kamera kolorowa, lewa)

    Args:
        calib_path: ścieżka do pliku kalibracji (.txt)

    Returns:
        Słownik {nazwa_macierzy: numpy array}, np. {"P2": array(3,4), "R_rect": array(3,3)}.
    """
    calib_path = Path(calib_path)
    calib = {}

    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(":", " ", 1).split()
            if len(parts) < 2:
                continue
            key = parts[0]
            numbers = np.array([float(x) for x in parts[1:]])
            
            if numbers.size == 12:
                calib[key] = numbers.reshape(3, 4)
            elif numbers.size == 9:
                calib[key] = numbers.reshape(3, 3)
            else:
                calib[key] = numbers
    
    return calib

def parse_tracking_labels(label_path):
    """Parsuje plik etykiet ground-truth w formacie KITTI tracking (label_02).

    Jeden plik odpowiada całej sekwencji (nie pojedynczej klatce) — każda linia
    zaczyna się od numeru klatki, do której należy.

    Struktura pojedynczej linii (kolumny rozdzielone spacją):

        [0]  frame        - numer klatki, do której należy ta etykieta
        [1]  track_id      - identyfikator obiektu (ten sam ID w kolejnych klatkach
                              oznacza ten sam, śledzony obiekt)
        [2]  type           - klasa obiektu: Car, Pedestrian, Cyclist, Van, Truck,
                              Tram, Misc, DontCare
        [3]  truncated      - 0.0-1.0, jak bardzo obiekt wychodzi poza kadr kamery
        [4]  occluded        - 0=widoczny, 1=częściowo zasłonięty, 2=mocno zasłonięty,
                              3=nieznany
        [5]  alpha            - kąt obserwacji obiektu względem kamery (radiany)
        [6:10] bbox           - 2D bounding box na obrazie: left, top, right, bottom
        [10:13] dimensions    - wymiary 3D: height, width, length (UWAGA na tę kolejność!)
        [13:16] location      - pozycja środka podstawy obiektu (x, y, z),
                              w układzie współrzędnych KAMERY, nie LiDAR-u
        [16]  rotation_y    - obrót wokół osi Y (pionowej) w układzie kamery, radiany
        [17]  score (opcjonalnie) - pewność detekcji; występuje tylko w plikach
                              wynikowych (predykcjach), nie w ground-truth

    Args:
        label_path: ścieżka do pliku label_02/<sequence>.txt

    Returns:
        Słownik {frame_id: [etykieta, ...]}, gdzie każda etykieta to dict
        z kluczami: track_id, type, bbox, dimensions, location, rotation_y.
    """
    
    label_path = Path(label_path)
    labels_by_frame = {}

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            frame_id = int(parts[0])
            entry = {
                "track_id": int(parts[1]),
                "type": parts[2],
                "bbox": [float(x) for x in parts[6:10]],
                "dimensions": [float(x) for x in parts[10:13]],
                "location": [float(x) for x in parts[13:16]],
                "rotation_y": float(parts[16]),
            }

            labels_by_frame.setdefault(frame_id, []).append(entry)

    return labels_by_frame