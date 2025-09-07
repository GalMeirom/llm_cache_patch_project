# /entrypoint.py
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path("/app")
SENTINEL = REPO / ".overlay_applied"

# Optional escape hatch: set SKIP_OVERLAY=1 to bypass overlay
if os.getenv("SKIP_OVERLAY", "0") not in ("1", "true", "True"):
    if not SENTINEL.exists():
        # Run exactly once
        subprocess.run(["python", "bootstrap/apply_overlay.py"], check=True)
        SENTINEL.write_text("ok\n")  # mark as applied

# Exec the requested command (default in Dockerfile if none)
cmd = sys.argv[1:] or ["pytest", "-q"]
os.execvp(cmd[0], cmd)
