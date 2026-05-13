PYTHON ?= python3

.PHONY: datasets learn test run all

datasets:
	$(PYTHON) prepare_datasets.py

benchmark:
	$(PYTHON) benchmark_experiments.py

learn:
	$(PYTHON) main.py --mode train

test:
	$(PYTHON) main.py --mode test

run:
	$(PYTHON) main.py --mode all

all: datasets learn test
