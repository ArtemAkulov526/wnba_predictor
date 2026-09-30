import os
from pathlib import Path

from wnba.paths import ENV_FILE
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


def _load_dotenv(path: Path = ENV_FILE) -> None:
    """Minimal .env loader so credentials stay out of source control.

    Deliberately dependency-free; existing environment variables win.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# psycopg2 decodes server error text using the client encoding; without this
# a failed connection on a non-UTF-8 Windows locale raises UnicodeDecodeError
# and hides the actual error.
os.environ.setdefault("PGCLIENTENCODING", "UTF8")

DATABASE_URL = os.getenv("WNBA_DB_URL") or (
    "postgresql+psycopg2://"
    f"{os.getenv('PGUSER', 'postgres')}:{os.getenv('PGPASSWORD', 'postgres')}"
    f"@{os.getenv('PGHOST', 'localhost')}:{os.getenv('PGPORT', '5432')}"
    f"/{os.getenv('PGDATABASE', 'postgres')}"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"client_encoding": "utf8"},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
