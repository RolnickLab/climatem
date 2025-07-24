from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"  # doesn't exist
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_PATH = PROJECT_ROOT / "configs"
PARAMS_PATH = PROJECT_ROOT / "params"  # doesn't exist
MAPPINGS_DIR = PROJECT_ROOT / "mappings"
TUNING_CONFIGS = CONFIGS_PATH / "tuning"  # doesn't exist

if sys.platform == "linux":
    SCRATCH_DIR = Path.home() / "scratch"
    print("Detected Linux system, using scratch directory: ", SCRATCH_DIR)
else:
    SCRATCH_DIR = PROJECT_ROOT.parent / "scratch"  # hardcoded for local machine
    print("Detected non-Linux system, using scratch directory: ", SCRATCH_DIR)
