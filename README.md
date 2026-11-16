# System automatycznej adnotacji danych LiDAR

Praca dyplomowa inżynierska — Politechnika Warszawska, Wydział Elektroniki i Technik Informacyjnych, Instytut Informatyki.

**Temat:** Projekt i implementacja systemu automatycznej adnotacji danych na potrzeby detekcji obiektów

**Autor:** Anna Wierzbik (nr albumu 331107)
**Promotor:** dr inż. Krystian Radlak

## Cel projektu

System wspierający przygotowanie danych treningowych dla modeli detekcji obiektów 3D, redukujący nakład pracy ręcznej poprzez automatyczne generowanie propozycji adnotacji (detekcja + tracking) i pozostawiający użytkownikowi rolę weryfikatora, a nie twórcy etykiet od zera.

Zbiór danych: **KITTI** (tracking benchmark + object detection benchmark).

## Architektura systemu

Pipeline składa się z pięciu modułów, uruchamianych sekwencyjnie dla każdej sekwencji danych:

```
Data Ingestion → Detection → Tracking & Association → Annotation Verification → Prediction Export
```

| Moduł | Katalog | Opis |
|---|---|---|
| Data Ingestion | `ingestion/` | Wczytywanie surowych chmur punktów, kalibracji i timestampów KITTI |
| Detection | `detection/` | Detekcja obiektów 3D (PointPillars, OpenPCDet), fine-tuning na KITTI |
| Tracking & Association | `tracking/` | Filtr Kalmana + metoda węgierska (tracking-by-detection) |
| Annotation Verification | `verification/`, `web/` | Sesja weryfikacji (decyzje, edycja, cofanie, interpolacja) i interfejs w przeglądarce |
| Prediction Export | `export/` | Eksport etykiet do formatu KITTI |

## Struktura repozytorium

```
lidar-auto-annotation/
├── ingestion/          # Data Ingestion Module
├── detection/           # Detection Module (PointPillars + fine-tuning)
├── tracking/             # Tracking & Association Module (Kalman + Hungarian)
├── verification/         # Annotation Verification Module (sesja, reguły edycji, cache)
├── web/                  # Interfejs w przeglądarce (FastAPI + three.js)
├── pipeline/             # Detekcja + tracking sekwencji, budowanie sesji
├── infra/                # Magazyn sesji i plików, kolejka zadań, ustawienia (LAL_*)
├── export/               # Prediction Export Module (format KITTI)
├── configs/              # Pliki konfiguracyjne (ścieżki, hiperparametry, progi)
├── scripts/              # Skrypty uruchomieniowe (np. run_pipeline.py)
├── tests/                # Testy jednostkowe modułów
├── data/                 # Dane KITTI (ignorowane przez git, patrz .gitignore)
├── requirements.txt
└── README.md
```

## Instalacja

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

OpenPCDet wymaga dodatkowej, osobnej instalacji (kompilacja rozszerzeń CUDA) — patrz [instrukcja OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md). Wymagany jest GPU z obsługą CUDA do treningu/fine-tuningu; sam pipeline (bez treningu) może działać na CPU, ale znacznie wolniej.

## Uruchomienie

```bash
python -m scripts.web                        # interfejs w przeglądarce: http://127.0.0.1:8000
python -m scripts.track_sequence --sequence 0000   # detekcja + tracking sekwencji KITTI do pliku cache
python -m scripts.export_labels --session results/sessions/<plik>.json --folder <folder z .bin>
```

Aplikacja desktopowa (okno Open3D) została usunięta; jedynym interfejsem jest przeglądarka
(`docs/adr/0009-remove-desktop-application.md`).

### Sesje w PostgreSQL (opcjonalnie)

Domyślnie sesje są plikami w `results/sessions/`. Baza PostgreSQL działa w Dockerze:

```bash
docker compose -f docker-compose.db.yml up -d          # start bazy (port 55432 na localhost)
export LAL_DATABASE_URL=postgresql+psycopg://lal:lal_local@127.0.0.1:55432/lal
python -m infra.migrate                                # migracje Alembic (schemat)
python -m scripts.import_sessions                      # jednorazowy import sesji z results/sessions/
LAL_SESSION_STORE=postgres python -m scripts.web       # serwer z sesjami w bazie
docker compose -f docker-compose.db.yml down           # stop (dane zostają; -v usuwa wolumen)
```

Hasło `lal_local` dotyczy tylko lokalnego kontenera (inne: zmienna `LAL_DB_PASSWORD` przy starcie).

### Testy

```bash
pytest                                  # testy jednostkowe
pytest tests/test_experiments           # testy eksperymentów
pytest -m postgres                      # testy z bazą (wymaga docker-compose.db.yml)
pytest -m e2e                           # testy w przeglądarce (requirements-dev.txt, Chromium)
pytest -m gpu                           # testy z prawdziwym modelem (GPU, OpenPCDet, dane KITTI)
```

## Dane

Pobierz zbiór KITTI (tracking benchmark) ze strony [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/) i umieść w `data/raw/` zgodnie ze strukturą opisaną w `data/raw/README.md`.

## Status projektu

Projekt w fazie realizacji (październik 2026 – styczeń 2027). Szczegółowy harmonogram i podział na zadania — zobacz tablicę Jira (projekt PI).

## Powiązane dokumenty

- Praca dyplomowa (rozdziały teoretyczne): patrz katalog dokumentacji pracy (poza repozytorium kodu).
