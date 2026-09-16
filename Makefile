PYTHON ?= python3

.PHONY: datasets learn test run all all_modes

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

RUN_TS  := $(shell date +%Y-%m-%d_%H-%M-%S)
RUN_DIR := outputs/$(RUN_TS)

all_modes:
	@mkdir -p $(RUN_DIR)
	@echo "Saving this run to $(RUN_DIR)"
	@echo "$(RUN_TS)" > $(RUN_DIR)/run_started_at.txt
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) prepare_datasets.py 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) main.py --multiseed 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) main.py --multiseed --congestion 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) main.py --multiseed --failure_rate 0.10 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) main.py --multiseed --topology watts_strogatz 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) main.py --multiseed --congestion --topology watts_strogatz 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) main.py --multiseed --failure_rate 0.10 --topology watts_strogatz 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) main.py --multiseed --topology erdos_renyi 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) main.py --multiseed --congestion --topology erdos_renyi 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) main.py --multiseed --failure_rate 0.10 --topology erdos_renyi 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) ablation.py --multiseed 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) xai_eval.py 2>&1 | tee -a $(RUN_DIR)/run.log
	GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON) scalability.py 2>&1 | tee -a $(RUN_DIR)/run.log
	@date +%Y-%m-%d_%H-%M-%S > $(RUN_DIR)/run_finished_at.txt
	@echo "Done. All results saved under $(RUN_DIR)"