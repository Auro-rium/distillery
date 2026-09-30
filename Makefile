PY ?= .venv/bin/python
PORT ?= 8000
IMAGE ?= distillery:local

.PHONY: dev test lint build docker dry-run

# Backend (uvicorn via the CLI, port $(PORT)) and Vite dev server, side by side.
# The app does not read .env itself, so this target sources it into the environment first.
# Ctrl-C stops both. Needs `pip install -e ".[dev]"` and `npm install` in frontend/.
dev:
	@set -a; if [ -f .env ]; then . ./.env; fi; set +a; \
	trap 'kill 0' INT TERM EXIT; \
	$(PY) -m distillery serve --port $(PORT) & \
	(cd frontend && npm run dev) & \
	wait

test:
	$(PY) -m pytest -q
	cd frontend && npm test

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy --strict src
	cd frontend && npm run typecheck

build:
	cd frontend && npm run build

docker:
	docker build -t $(IMAGE) .

# OFFLINE and FAKE: fake models, fake prices, no network. Proves plumbing only, not model quality.
dry-run:
	$(PY) -m distillery run --pack sql --scale tiny --dry-run
