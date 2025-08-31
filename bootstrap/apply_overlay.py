from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root
BOOTSTRAP = ROOT / "bootstrap"
OVERLAY_ROOT = ROOT / "overlays"  # <-- singular
# overlay sources:
SRC_MEMORY = OVERLAY_ROOT / "gptcache" / "manager" / "eviction" / "memory_cache.py"
SRC_POLICIES = OVERLAY_ROOT / "gptcache" / "manager" / "eviction" / "policies"

ARTIFACTS = BOOTSTRAP / "artifacts" / "overlays"
MANIFEST = BOOTSTRAP / "MANIFEST.json"


def _gptcache_root() -> Path:
    import gptcache

    return Path(gptcache.__file__).resolve().parent


def main() -> int:
    if not SRC_MEMORY.exists():
        print(f"[apply] ERROR: missing {SRC_MEMORY}", file=sys.stderr)
        return 2

    gpt_root = _gptcache_root()
    dst_memory = gpt_root / "manager" / "eviction" / "memory_cache.py"
    dst_pols = gpt_root / "manager" / "eviction" / "policies"

    if not dst_memory.exists():
        print(f"[apply] ERROR: destination not found: {dst_memory}", file=sys.stderr)
        return 3

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_memory = ARTIFACTS / f"memory_cache.py.backup-{stamp}"

    # backup + apply file
    shutil.copy2(dst_memory, backup_memory)
    shutil.copy2(SRC_MEMORY, dst_memory)
    print(f"[apply] backup: {dst_memory} -> {backup_memory}")
    print(f"[apply] file  : {SRC_MEMORY} -> {dst_memory}")

    # replace policies dir (if provided in overlay)
    created_policies = False
    if SRC_POLICIES.exists():
        if dst_pols.exists():
            shutil.rmtree(dst_pols)
        shutil.copytree(SRC_POLICIES, dst_pols)
        created_policies = True
        print(f"[apply] dir   : {SRC_POLICIES} -> {dst_pols}")

    # write manifest
    ops = [{"kind": "restore_file", "dest": str(dst_memory), "backup": str(backup_memory)}]
    if created_policies:
        ops.append({"kind": "remove_dir", "path": str(dst_pols)})

    MANIFEST.write_text(
        json.dumps({"gptcache_root": str(gpt_root), "ops": ops}, indent=2), encoding="utf-8"
    )
    print(f"[apply] manifest: {MANIFEST}")
    print("[apply] DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
