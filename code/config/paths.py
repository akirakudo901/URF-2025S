from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env once, safely
load_dotenv()

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
DATA_DIR = Path(os.environ["DATA_DIR"]).resolve() # separates code in home vs. data in projects for Compute Canada

SRC_DIR = PROJECT_ROOT / "code"
