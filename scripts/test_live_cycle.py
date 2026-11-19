import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from studio.autonomous_live_clipper import run_autonomous_clipping_cycle

if __name__ == "__main__":
    run_autonomous_clipping_cycle()
