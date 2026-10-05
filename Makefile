# GloXRoute — run everything from the project root:  make all
PYTHON ?= python3
SHELL  := /bin/bash
.SHELLFLAGS := -eo pipefail -c        # stop on the first failing step (tee no longer hides errors)

RUN_TS  := $(shell date +%Y-%m-%d_%H-%M-%S)
RUN_DIR ?= outputs/$(RUN_TS)
RUN      = GLOXROUTE_RUN_DIR=$(RUN_DIR) $(PYTHON)
LOG      = 2>&1 | tee -a $(RUN_DIR)/run.log

.PHONY: all all_modes datasets calibrate matrix dynamic ablation xai scalability paper figures clean

## Full pipeline: experiments + paper CSVs + figures/tables
all: all_modes paper

## Experiments only (9-config matrix + ablation + XAI + scalability)
all_modes: datasets calibrate matrix ablation xai scalability
	@date +%Y-%m-%d_%H-%M-%S > $(RUN_DIR)/run_finished_at.txt
	@echo "Done. All results saved under $(RUN_DIR)"

$(RUN_DIR):
	@mkdir -p $(RUN_DIR)
	@echo "$(RUN_TS)" > $(RUN_DIR)/run_started_at.txt
	@echo "Saving this run to $(RUN_DIR)"

datasets: | $(RUN_DIR)
	$(RUN) -m scripts.prepare_datasets $(LOG)

## GloXRoute cost-term weight, chosen on training graphs only
calibrate: | $(RUN_DIR)
	$(RUN) -m experiments.calibrate $(LOG)

matrix: | $(RUN_DIR)
	@for topo in barabasi_albert watts_strogatz erdos_renyi; do \
	  for cond in "" "--congestion" "--failure_rate 0.10"; do \
	    echo ">>> topology=$$topo condition=$${cond:-normal}"; \
	    $(RUN) main.py --multiseed --topology $$topo $$cond $(LOG); \
	  done; \
	done

## Optional (not part of the paper pipeline): time-varying loads
dynamic: | $(RUN_DIR)
	@for topo in barabasi_albert watts_strogatz erdos_renyi; do \
	  echo ">>> topology=$$topo condition=dynamic"; \
	  $(RUN) main.py --multiseed --topology $$topo --dynamic $(LOG); \
	done

ablation: | $(RUN_DIR)
	$(RUN) -m experiments.ablation --multiseed $(LOG)

xai: | $(RUN_DIR)
	$(RUN) -m experiments.xai_eval $(LOG)

scalability: | $(RUN_DIR)
	$(RUN) -m experiments.scalability $(LOG)

## Collect paper CSVs from the run and rebuild all figures and tables
paper:
	bash scripts/gather_paper_csvs.sh $(RUN_DIR) paper_data
	$(PYTHON) make_figures_tables.py --data paper_data --out figs

## Figures/tables only, from an existing paper_data/
figures:
	$(PYTHON) make_figures_tables.py --data paper_data --out figs

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
