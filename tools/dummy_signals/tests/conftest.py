import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT, ROOT / "third_party" / "consistency"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
