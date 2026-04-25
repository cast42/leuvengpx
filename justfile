set dotenv-load

default:
    just --list

install:
    uv sync
    npm install

update:
    uv sync --upgrade
    npm update

clean:
    rm -rf dist node_modules .pytest_cache .ruff_cache .ty .venv

generate:
    uv run python -m src.generate_site

build: generate
    npm run build

preview: generate
    npm run dev -- --host 127.0.0.1

serve: build
    npm run preview -- --host 127.0.0.1

add-gpx file:
    uv run python -m src.add_gpx "{{file}}"
    just generate

publish: check
    uv run python scripts/trigger_pages_deploy.py

lint:
    uv run ruff check --fix
    uv run ruff format
    npm run lint

typing:
    uv run ty check

test:
    uv run pytest -q
    npm test

check: lint typing test build
