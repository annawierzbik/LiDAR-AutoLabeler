"""
Prediction Export Module
==========================

Konwertuje wewnętrzną reprezentację adnotacji (detekcje + tracking + poprawki
użytkownika) do formatu etykiet zgodnego ze zbiorem KITTI, oraz waliduje
poprawność wygenerowanych plików.

Planowana zawartość modułu:
- kitti_writer.py   -> zapis etykiet do formatu KITTI (.txt per klatka)
- validate.py         -> walidacja zgodności z KITTI devkit

Powiązane taski Jira: faza-6 (PI-35, PI-36).
"""
