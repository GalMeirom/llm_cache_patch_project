# bootstrap/install_and_verify.py
"""
Install (if needed) and verify GPTCache import, version, and paths.
- Prints: gptcache version, package root, site-packages.
- Writes a JSON log under artifacts/logs/install_and_verify.json
- Exits with nonzero status if verification fails.
"""

from __future__ import annotations

import json
import sys
import sysconfig
from pathlib import Path

LOG_DIR = Path("artifacts/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "install_and_verify.json"


def main() -> int:
    info = {"ok": False, "error": None}
    try:
        purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
        info["purelib"] = str(purelib)

        try:
            import gptcache  # type: ignore
        except Exception as e:  # not installed or import error
            info["error"] = f"import error: {e}"
            # Suggest next steps on stdout, but still write the log:
            print("GPTCache not importable. Try:", file=sys.stderr)
            print("  pip install -r requirements.txt", file=sys.stderr)
            _write_log(info)
            return 2

        pkg_file = Path(gptcache.__file__).resolve()
        pkg_root = pkg_file.parent
        version = getattr(gptcache, "__version__", "unknown")

        info.update(
            {
                "ok": True,
                "version": version,
                "package_file": str(pkg_file),
                "package_root": str(pkg_root),
            }
        )

        print("Verified GPTCache:")
        print(f"  version       : {version}")
        print(f"  package_root  : {pkg_root}")
        print(f"  package_file  : {pkg_file}")
        print(f"  site-packages : {purelib}")

        _write_log(info)
        return 0

    except Exception as e:
        info["error"] = f"unexpected failure: {e}"
        _write_log(info)
        print(f"[install_and_verify] ERROR: {e}", file=sys.stderr)
        return 1


def _write_log(obj: dict) -> None:
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text(json.dumps(obj, indent=2))
    except Exception as _:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
