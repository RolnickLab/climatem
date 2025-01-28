from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_PATH = SCRIPTS_DIR / "configs"
PARAMS_PATH = SCRIPTS_DIR / "params"
MAPPINGS_DIR = PROJECT_ROOT / "climatem" / "mappings"
