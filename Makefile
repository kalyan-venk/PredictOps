.PHONY: install lint test train drift register serve pipeline docker-build docker-up

install:
	pip install -r requirements.txt
	pip install -e .

lint:
	ruff check .

test:
	pytest -v

train:
	python -m predictops.train

drift:
	python -m predictops.drift

register:
	python -m predictops.registry

serve:
	uvicorn predictops.app:app --host 0.0.0.0 --port 8000

# Full loop: load -> train -> eval gate -> drift check -> log to MLflow -> serve.
pipeline: train test drift register serve

docker-build:
	docker build -t predictops:local .

docker-up:
	docker-compose up
