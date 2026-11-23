.PHONY: setup data train api web dev test e2e clean

setup:
	pip install --break-system-packages -r api/requirements.txt -r training/requirements.txt
	cd web && npm ci

data:
	python3 training/fetch_data.py

train:
	python3 training/train.py

api:
	cd api && uvicorn app.main:app --reload --port 8000

web:
	cd web && NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run dev

test:
	python3 -m pytest tests -q

build:
	cd web && NEXT_PUBLIC_API_BASE=http://localhost:8000 npm run build

clean:
	rm -rf web/.next web/out api/**/__pycache__
