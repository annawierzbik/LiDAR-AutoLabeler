"""Browser tabs of the API v1 (docs/adr/0017): which user opened which sequence in which tab, and
the job that builds the session. The binding lives in the `tabs` table, so any API replica (and an
API restarted meanwhile) serves the tab; the session key and size come from the job's result."""
from __future__ import annotations

import uuid

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Engine

from infra.db import TabRow

COLUMNS = (TabRow.id, TabRow.user_id, TabRow.client_id, TabRow.sequence_id, TabRow.job_id, TabRow.title,
           TabRow.status)


class TabStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:16]

    def create(self, tab_id: str, user_id: int, client_id: str, sequence_id: int, job_id: str, title: str) -> None:
        with self.engine.begin() as c:
            c.execute(insert(TabRow).values(id=tab_id, user_id=user_id, client_id=client_id, sequence_id=sequence_id,
                                            job_id=job_id, title=title, status="open"))

    def get(self, tab_id: str, user_id: int | None) -> dict:
        """KeyError if there is no such tab or (with `user_id`) it is another user's."""
        q = select(*COLUMNS).where(TabRow.id == tab_id)
        if user_id is not None:
            q = q.where(TabRow.user_id == user_id)
        with self.engine.connect() as c:
            row = c.execute(q).first()
        if row is None:
            raise KeyError(tab_id)
        return dict(row._mapping)

    def finish(self, tab_id: str) -> None:
        with self.engine.begin() as c:
            c.execute(update(TabRow).where(TabRow.id == tab_id).values(status="finished"))
