import sys
from pathlib import Path

# Project root (CosyVoice/)
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
