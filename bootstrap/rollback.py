from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "bootstrap"
MANIFEST = BOOTSTRAP / "MANIFEST.json"


def main() -> int:
    if not MANIFEST.exists():
        print("[rollback] ERROR: MANIFEST.json not found", file=sys.stderr)
        return 2

    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ops = data.get("ops", [])

    # reverse the ops
    for op in reversed(ops):
        kind = op.get("kind")
        if kind == "restore_file":
            dest = Path(op["dest"])
            backup = Path(op["backup"])
            if backup.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, dest)
                print(f"[rollback] restored: {dest} <- {backup}")
            else:
                print(f"[rollback] WARNING: missing backup {backup}")
        elif kind == "remove_dir":
            p = Path(op["path"])
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
                print(f"[rollback] removed dir: {p}")

    try:
        MANIFEST.unlink()
        print("[rollback] removed MANIFEST.json")
    except Exception:
        pass

    print("[rollback] DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
