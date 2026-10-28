"""Safe CSV writes for the logs and archives in data/."""
import os
import time
from pathlib import Path

import pandas as pd


def write_csv(df: pd.DataFrame, path: Path, retries: int = 20, wait: float = 0.5) -> None:
    """Write via a temp file and an atomic rename.

    A crash mid-save cannot leave a half-written file. OneDrive (or an editor
    with the file open) briefly locks files while syncing them, so the rename
    is retried for up to ~10 seconds before giving up.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    for attempt in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(wait)
