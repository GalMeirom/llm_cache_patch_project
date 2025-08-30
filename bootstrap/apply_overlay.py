from __future__ import annotations

import hashlib
import json
import shutil
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY_SRC = ROOT / "overlays" / "gptcache"
ARTIFACTS = ROOT / "artifacts" / "overlays"
BACKUPS = ARTIFACTS / "backups"
MANIFEST = ROOT / "overlays" / "MANIFEST.json"

MARK_BEGIN = "# BEGIN LLM_CACHE_PATCH"
MARK_END = "# END LLM_CACHE_PATCH"
INJECT_BLOCK = f"""{MARK_BEGIN}
# This block is injected by llm_cache_project overlay.
from ._llm_patch_probe import LLM_PATCH_SENTINEL as _LLM_PATCH_SENTINEL  # noqa: F401
LLM_PATCH_ACTIVE = True
{MARK_END}
"""


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_gptcache_root() -> Path:
    try:
        import gptcache  # type: ignore
    except Exception as e:
        print(f"[apply_overlay] ERROR: cannot import gptcache: {e}", file=sys.stderr)
        sys.exit(2)
    pkg_file = Path(gptcache.__file__).resolve()
    return pkg_file.parent


def inject_block(init_py: Path, manifest_files: list, backup_dir: Path):
    text = init_py.read_text(encoding="utf-8", errors="ignore")
    if MARK_BEGIN in text and MARK_END in text:
        print("[apply_overlay] injection already present, skipping.")
        return

    # backup original __init__.py if not already backed up
    backup_target = backup_dir / "gptcache" / "__init__.py"
    backup_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(init_py, backup_target)

    before_hash = sha256_of(init_py)
    with init_py.open("a", encoding="utf-8") as f:
        f.write("\n\n" + INJECT_BLOCK + "\n")

    manifest_files.append(
        {
            "dest": str(init_py),
            "action": "modified",
            "backed_up_to": str(backup_target),
            "sha256_before": before_hash,
        }
    )
    print(f"[apply_overlay] injected marker into {init_py}")


def copy_overlay_files(overlay_src: Path, dst_root: Path, manifest_files: list, backup_dir: Path):
    if not overlay_src.exists():
        print(f"[apply_overlay] no overlay dir at {overlay_src}, skipping file copy.")
        return
    for src in overlay_src.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(overlay_src)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            # backup then overwrite
            backup_path = backup_dir / "gptcache" / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, backup_path)
            before = sha256_of(dst)
            shutil.copy2(src, dst)
            manifest_files.append(
                {
                    "dest": str(dst),
                    "action": "overwritten",
                    "backed_up_to": str(backup_path),
                    "sha256_before": before,
                }
            )
            print(f"[apply_overlay] overwritten {dst} (backup saved)")
        else:
            shutil.copy2(src, dst)
            manifest_files.append(
                {
                    "dest": str(dst),
                    "action": "created",
                }
            )
            print(f"[apply_overlay] created {dst}")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    BACKUPS.mkdir(parents=True, exist_ok=True)

    gpt_root = find_gptcache_root()
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BACKUPS / ts

    manifest = {
        "applied_at": ts,
        "purelib": str(purelib),
        "gptcache_root": str(gpt_root),
        "backup_dir": str(backup_dir),
        "files": [],
    }

    backup_dir.mkdir(parents=True, exist_ok=True)

    # 1) copy overlay files
    copy_overlay_files(OVERLAY_SRC, gpt_root, manifest["files"], backup_dir)

    # 2) inject block into __init__.py
    init_py = gpt_root / "__init__.py"
    inject_block(init_py, manifest["files"], backup_dir)

    # 3) write manifest
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(" - Probe file copied, marker injected.")
    print(
        " - Verify with:  python -c "
        "\"import gptcache; print(getattr(gptcache,'LLM_PATCH_ACTIVE',None))\""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
