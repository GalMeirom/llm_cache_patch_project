# LLM Cache Project

Approach: install-then-patch GPTCache to add SLRU, TinyLFU+SLRU, and GDSR.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
pre-commit install

Minor: validate CI pipeline.