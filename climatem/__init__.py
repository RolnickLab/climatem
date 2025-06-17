from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"  # doesn't exist
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_PATH = PROJECT_ROOT / "configs"
PARAMS_PATH = PROJECT_ROOT / "params"  # doesn't exist
MAPPINGS_DIR = PROJECT_ROOT / "mappings"
TUNING_CONFIGS = CONFIGS_PATH / "tuning"  # doesn't exist