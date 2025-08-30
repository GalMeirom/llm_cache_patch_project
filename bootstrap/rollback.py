from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "overlays" / "MANIFEST.json"

MARK_BEGIN = "# BEGIN LLM_CACHE_PATCH"
MARK_END = "# END LLM_CACHE_PATCH"
INJECT_RE = re.compile(
    r"\n?\s*# BEGIN LLM_CACHE_PATCH.*?# END LLM_CACHE_PATCH\s*\n?",
    re.DOTALL,
)


def remove_injected_block(init_py: Path):
    txt = init_py.read_text(encoding="utf-8", errors="ignore")
    new = INJECT_RE.sub("", txt)
    if new != txt:
        init_py.write_text(new, encoding="utf-8")
        print(f"[rollback] removed injected block from {init_py}")
    else:
        print(f"[rollback] no injected block found in {init_py}")


def main() -> int:
    if not MANIFEST.exists():
        print("[rollback] ERROR: no MANIFEST.json; nothing to rollback.", file=sys.stderr)
        return 2

    m = json.loads(MANIFEST.read_text())
    files = m.get("files", [])
    # restore in reverse order (safer)
    for entry in reversed(files):
        dest = Path(entry["dest"])
        action = entry["action"]
        if action in {"overwritten", "modified"}:
            backup = Path(entry["backed_up_to"])
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                shutil.copy2(backup, dest)
                print(f"[rollback] restored {dest} from backup")
            else:
                print(f"[rollback] WARNING: backup missing for {dest}: {backup}")
        elif action == "created":
            if dest.exists():
                dest.unlink()
                print(f"[rollback] deleted created file {dest}")

    # also ensure the injected block is removed from __init__.py
    init_py = Path(m["gptcache_root"]) / "__init__.py"
    if init_py.exists():
        remove_injected_block(init_py)

    print("[rollback] SUCCESS. Overlay reverted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
