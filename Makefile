PYTHON = poetry run python

DIR = data/test-build-0

export PYTHONPATH = $(shell pwd)

setup:
	poetry install --no-root

clean:
	rm -rf $(DIR) || true
	rm -rf dev/test-case/ggml || true
	rm -rf dev/test-case/build || true
	rm -rf test_data/* || true
	rm -rf models/* || true

test:
	$(PYTHON) -m pytest -vv tests

test_case:
	cd dev/test-case && git clone https://github.com/ggerganov/ggml.git && mkdir build && cd build && cmake .. && make && ./torch_jit

measure_performance:
	for num_blocks in 4 8 16 32; do \
		for input_dim in 128 256 512 1024; do \
		  $(PYTHON) -m scripts.measure_performance --num-blocks $$num_blocks --input-dim $$input_dim; \
		done; \
	done
