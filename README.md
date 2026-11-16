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
| Annotation Verification | `verification/`, `web/`, `api/` | Sesja weryfikacji (decyzje, edycja, cofanie, interpolacja) i interfejs w przeglądarce |
| Prediction Export | `export/` | Eksport etykiet do formatu KITTI |

## Struktura repozytorium

```
lidar-auto-annotation/
├── ingestion/          # Data Ingestion Module
├── detection/           # Detection Module (PointPillars + fine-tuning)
├── tracking/             # Tracking & Association Module (Kalman + Hungarian)
├── verification/         # Annotation Verification Module (sesja, reguły edycji, cache)
├── api/                  # API v1 (FastAPI): logowanie, sekwencje, karty, zadania, health
├── web/                  # Strony w przeglądarce (three.js) i ReviewService
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

### Co jest potrzebne do czego

**Aplikacja WWW** (API i strony) wymaga zawsze: PostgreSQL (użytkownicy, sekwencje, karty, sesje,
zadania), Redisa (broker kolejki), co najmniej jednego workera kolejki `gpu` (otwieranie sesji z
detekcją) i jednego workera kolejki `cpu` (sesje ręczne, eksport) oraz ustawień
`LAL_SESSION_STORE=postgres`, `LAL_JOB_QUEUE=celery` i `LAL_JWT_SECRET`. Bez nich serwer nie
startuje.

**Skrypty działające tylko na plikach lokalnych** (bez bazy, Redisa i workerów):

| skrypt | co robi |
|---|---|
| `scripts.detect_frame` | detekcja na jednej klatce KITTI |
| `scripts.track_sequence` | detekcja + tracking sekwencji KITTI do pliku cache w `results/` |
| `scripts.evaluate_sequence`, `scripts.evaluate_all` | ewaluacja względem ground truth |
| `scripts.export_labels` | eksport etykiet z **pliku** sesji JSON (sesje z bazy eksportuje się w przeglądarce) |
| `scripts.verify_dataset` | sprawdzenie struktury danych KITTI |

**Skrypty wymagające PostgreSQL:** `scripts.create_user`, `scripts.import_sessions` (czyta pliki
sesji, zapisuje do bazy), `scripts.assign_session_owner`, `scripts.bench_session_store` (porównuje magazyn plikowy z bazą).
`FileSessionStore` (sesje jako pliki JSON) jest używany już tylko przez `scripts.bench_session_store`
i testy jednostkowe.

```bash
python -m scripts.track_sequence --sequence 0000   # detekcja + tracking sekwencji KITTI do pliku cache
python -m scripts.export_labels --session results/sessions/<plik>.json --folder <folder z .bin>
```

Aplikacja desktopowa (okno Open3D) została usunięta; jedynym interfejsem jest przeglądarka
(`docs/adr/0009-remove-desktop-application.md`).

### Interfejs w przeglądarce (API v1)

Serwer WWW zawsze korzysta z PostgreSQL, Redisa i workerów Celery (oba serwisy są w
`docker-compose.db.yml`). Sesja jest budowana przez workera, eksport też jest zadaniem kolejki.
Na Windows workery działają z pulą `solo`:

```bash
docker compose -f docker-compose.db.yml up -d          # PostgreSQL (port 55432) i Redis (port 56379)
export LAL_DATABASE_URL=postgresql+psycopg://lal:lal_local@127.0.0.1:55432/lal
export LAL_REDIS_URL=redis://127.0.0.1:56379/0
export LAL_SESSION_STORE=postgres LAL_JOB_QUEUE=celery
export LAL_JWT_SECRET=<losowy ciąg, co najmniej 32 znaki>
export LAL_COOKIE_SECURE=false                         # tylko lokalnie, bez TLS (http)
export LAL_DATA_ROOT=/dane/lidar                       # foldery, które admin może zarejestrować
python -m infra.migrate                                # migracje Alembic (schemat)
python -m scripts.create_user admin --role admin       # pierwszy administrator (pyta o hasło)
celery -A infra.celery_app worker -Q gpu -P solo -n gpu@%h    # otwieranie sesji z detekcją (jeden model)
celery -A infra.celery_app worker -Q cpu -P solo -n cpu@%h    # eksport, sesje ręczne (można uruchomić kilka)
python -m scripts.web                                  # http://127.0.0.1:8000, logowanie
docker compose -f docker-compose.db.yml down           # stop (dane zostają; -v usuwa wolumen)
```

Hasło `lal_local` dotyczy tylko lokalnego kontenera (inne: zmienna `LAL_DB_PASSWORD` przy starcie).
Workery potrzebują tych samych zmiennych `LAL_*` co serwer.

- Logowanie: role `annotator` i `admin`; hasła jako skróty argon2. Tokeny JWT są wyłącznie w
  ciasteczkach HttpOnly, żądania zmieniające stan niosą nagłówek CSRF
  (`docs/adr/0016-auth-tokens-and-csrf.md`). Kolejnych użytkowników dodaje admin
  (`POST /api/v1/users`) lub `scripts.create_user`.
- Sekwencje: użytkownik wgrywa archiwum `.zip` z plikami `.bin` (limit rozmiaru, bezpieczne
  rozpakowanie), admin rejestruje folder leżący pod `LAL_DATA_ROOT` dla wybranego użytkownika
  (`docs/adr/0003-sequence-sources.md`). Wpisywanie dowolnej ścieżki zostało usunięte.
  Wgraną sekwencję usuwa jej właściciel lub admin (`DELETE /api/v1/sequences/<id>`); gdy ma
  sesje, tylko admin z `force=true` (sesje też są usuwane). Usuwane są klatki i wpisy cache detekcji.
- Każdy użytkownik widzi tylko swoje sekwencje, sesje i zadania; admin widzi wszystko
  (`docs/adr/0017-ownership-tabs-and-session-locks.md`).
- Dokumentacja API: `/api/v1/docs`; stan: `/healthz` i `/readyz`
  (`docs/adr/0018-api-v1-conventions.md`). Serwer nie trzyma stanu między żądaniami, więc kilka
  procesów API może działać za load balancerem.

Sesje zapisane wcześniej w plikach można przenieść do bazy:
`python -m scripts.import_sessions --owner <użytkownik>`. Sesje bez właściciela, które już są w
bazie (widzi je tylko admin), przypisuje `python -m scripts.assign_session_owner <użytkownik> --all`
(`--list` pokazuje je, `--dry-run` tylko wypisuje nowe klucze). Potem otwiera je karta tego
użytkownika dla tej samej sekwencji (`docs/adr/0017-ownership-tabs-and-session-locks.md`).

### Testy

```bash
pytest                                  # testy jednostkowe
pytest tests/test_experiments           # testy eksperymentów
pytest -m postgres                      # testy z bazą, w tym API v1 (wymaga docker-compose.db.yml)
pytest -m e2e                           # testy w przeglądarce: API, worker, baza, Redis (requirements-dev.txt, Chromium)
pytest -m worker                        # prawdziwe procesy workerów Celery (Redis + PostgreSQL)
pytest -m gpu                           # testy z prawdziwym modelem (GPU, OpenPCDet, dane KITTI)
```

## Dane

Pobierz zbiór KITTI (tracking benchmark) ze strony [cvlibs.net/datasets/kitti](https://www.cvlibs.net/datasets/kitti/) i umieść w `data/raw/` zgodnie ze strukturą opisaną w `data/raw/README.md`.

## Status projektu

Projekt w fazie realizacji (październik 2026 – styczeń 2027). Szczegółowy harmonogram i podział na zadania — zobacz tablicę Jira (projekt PI).

## Powiązane dokumenty

- Praca dyplomowa (rozdziały teoretyczne): patrz katalog dokumentacji pracy (poza repozytorium kodu).
