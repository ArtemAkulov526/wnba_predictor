"""Project paths, resolved from this file so they work regardless of cwd."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"          # logs, archives, odds snapshots (gitignored)
MODELS_DIR = PROJECT_ROOT / "models"      # trained artefacts (gitignored)
TEMPLATES_DIR = Path(__file__).resolve().parent / "web" / "templates"
