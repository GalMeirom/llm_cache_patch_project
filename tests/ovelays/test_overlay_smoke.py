import os
import subprocess
import sys

import pytest

RUN = os.environ.get("RUN_OVERLAY_TESTS") == "1"


@pytest.mark.skipif(
    not RUN, reason="overlay smoke test disabled; set RUN_OVERLAY_TESTS=1 to enable"
)
def test_overlay_apply_and_rollback():
    # fresh python processes to observe package import behavior
    def py(cmd):
        return subprocess.check_output([sys.executable, "-c", cmd], text=True).strip()

    # apply overlay
    subprocess.check_call([sys.executable, "bootstrap/apply_overlay.py"])

    out = py("import gptcache; print(getattr(gptcache, 'LLM_PATCH_ACTIVE', None))")
    assert out == "True"

    # rollback overlay
    subprocess.check_call([sys.executable, "bootstrap/rollback.py"])

    out2 = py(
        "import importlib, gptcache; importlib.reload(gptcache); "
        "print(hasattr(gptcache, 'LLM_PATCH_ACTIVE'))"
    )
    assert out2 == "False"
