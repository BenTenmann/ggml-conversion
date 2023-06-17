PYTHON = poetry run python

DIR = data/test-build-0

export PYTHONPATH = $(shell pwd)

setup:
	poetry install --no-root

clean:
	rm -rf $(DIR) || true
	rm -rf dev/test-case/ggml || true
	rm -rf dev/test-case/build || true

test:
	$(PYTHON) -m pytest -vv tests -k "test_model_builds"

test_case:
	cd dev/test-case && git clone https://github.com/ggerganov/ggml.git && mkdir build && cd build && cmake .. && make && ./torch_jit
