"""
Detection Module
=================

Odpowiada za detekcję obiektów 3D w pojedynczej klatce chmury punktów.
Bazuje na pretrenowanym modelu PointPillars (biblioteka OpenPCDet),
doszkalanym metodą fine-tuningu na wybranych sekwencjach KITTI.

Planowana zawartość modułu:
- detector.py    -> wrapper Detector.predict(frame) -> List[BoundingBox3D]
- finetune.py     -> skrypt fine-tuningu modelu na danych KITTI
- evaluate.py     -> ewaluacja mAP przed/po fine-tuningu

Powiązane taski Jira: faza-2 (PI-18 ... PI-22).
"""
