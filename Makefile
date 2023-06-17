PYTHON = poetry run python

DIR = data/test-build-0

export PYTHONPATH = $(shell pwd)

setup:
	poetry install --no-root

clean:
	rm -rf $(DIR) || true

test:
	$(PYTHON) -m pytest -vv tests -k "test_model_builds[AddConst]"
