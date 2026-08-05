"""SQLAlchemy engine / session management."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, inspect
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from swing_trader.config import get_settings


class Base(DeclarativeBase):
    pass


def _unwrap_numpy_scalars(session: Session, flush_context, instances) -> None:
    """`before_flush` hook: unwrap numpy scalar types (`np.float64`,
    `np.int64`, `np.bool_`, ...) on every mapped column attribute of every
    new/dirty ORM object, for every `Session` anywhere in this codebase
    (registered globally on the `Session` class below, not on one
    particular sessionmaker).

    Why globally, not at each call site: pandas/numpy computations
    routinely hand back numpy scalars instead of plain Python
    floats/ints/bools -- values pulled from a DataFrame column, `.mean()`,
    a `Series.get(...)`, a numpy array element, etc. SQLite (used in tests)
    tolerates this transparently, but psycopg2/PostgreSQL has no default
    adapter for numpy scalar types, so `session.add(SomeOrmObject(x=np.float64(1.0)))`
    followed by a flush fails with `InvalidSchemaName: schema "np" does not
    exist` -- it literally renders `np.float64(...)` into the SQL text.
    This bug has already hit `StockFeature` (features/engineering.py) and
    `RegimeHistory` (models/regime_detector.py, fed numpy-typed
    vix/spy_adx/sector_breadth_pct values from a pandas Series in
    models/pipeline.py's `build_context_from_db`). Rather than auditing
    and `float()`-wrapping every individual place a pandas/numpy-derived
    value might end up on an ORM attribute -- an easy thing to miss again
    in new code -- this hook catches every case at flush time, once.
    """
    for obj in list(session.new) + list(session.dirty):
        mapper = inspect(obj).mapper
        for attr in mapper.column_attrs:
            value = getattr(obj, attr.key, None)
            if value is None or isinstance(value, (str, bytes)):
                continue
            if hasattr(value, "item"):
                try:
                    setattr(obj, attr.key, value.item())
                except (ValueError, AttributeError, TypeError):
                    pass


# Registered on the `Session` class itself (not a specific sessionmaker) so
# it applies to every session in the process, including test fixtures that
# build their own sessionmaker (see tests/conftest.py).
event.listen(Session, "before_flush", _unwrap_numpy_scalars)


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope: `with session_scope() as db: ...`."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
