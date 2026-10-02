"""
Tracking & Association Module
==============================

Realizuje paradygmat tracking-by-detection: propaguje adnotacje w czasie,
utrzymując spójne identyfikatory (ID) obiektów pomiędzy klatkami.

Planowana zawartość modułu:
- kalman_filter.py   -> estymacja stanu obiektu (pozycja, wymiary, orientacja, prędkość)
- association.py      -> macierz kosztów (3D IoU) + algorytm węgierski
- tracklet.py          -> zarządzanie ścieżkami obiektów (tworzenie/aktualizacja/usuwanie)
- tracker.py           -> główna pętla: predykcja -> dopasowanie -> aktualizacja

Powiązane taski Jira: faza-3 (PI-23 ... PI-27), faza-4 (PI-28 ... PI-30).
"""
