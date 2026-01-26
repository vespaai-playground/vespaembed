
.PHONY: quality style test

quality:
	black --check .
	isort --check-only .
	flake8 .

style:
	black .
	isort .

test:
	pytest -sv ./src/

pip:
	rm -rf build/
	rm -rf dist/
	python -m build
	twine upload dist/* --verbose