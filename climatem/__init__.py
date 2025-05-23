from pathlib import Path
import os

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data") # doesn't exist
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
CONFIGS_PATH = os.path.join(PROJECT_ROOT, "configs")
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params") # doesn't exist
MAPPINGS_DIR = os.path.join(PROJECT_ROOT, "mappings")
TUNING_CONFIGS = os.path.join(CONFIGS_PATH, "tuning") # doesn't exist
