.PHONY: setup data train test clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

data:
	.venv/bin/python scripts/download_data.py

train:
	.venv/bin/python -m mlgroup8.train --config configs/tiny.yaml

test:
	.venv/bin/python -m pytest -q

clean:
	rm -rf .pytest_cache runs
