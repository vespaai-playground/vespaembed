
.PHONY: quality style test frontend-build pip

quality:
	black --check .
	isort --check-only .
	flake8 .

style:
	black .
	isort .

test:
	pytest -sv ./src/

frontend-build:
	cd frontend && npm ci && npm run build

pip: frontend-build
	rm -rf build/
	rm -rf dist/
	python -m build
	twine upload dist/* --verbose
