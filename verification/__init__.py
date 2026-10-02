"""
Annotation Verification Module
================================

Interfejs użytkownika do weryfikacji automatycznie wygenerowanych adnotacji.
Dwa tryby pracy: weryfikacja (klatka po klatce, accept/reject/edit) oraz
playback demo (automatyczne odtwarzanie sekwencji z ramkami 3D w czasie
rzeczywistym, do celów prezentacyjnych).

Planowana zawartość modułu:
- viewer.py       -> wizualizacja chmury punktów + ramek 3D + ID (Open3D)
- verify_mode.py   -> tryb weryfikacji: accept/reject/edit + zapis poprawek
- playback_mode.py -> tryb automatycznego odtwarzania sekwencji (demo)

Powiązane taski Jira: faza-4 (PI-31), faza-5 (PI-32 ... PI-34).
"""
