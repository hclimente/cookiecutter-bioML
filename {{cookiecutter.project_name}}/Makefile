.PHONY: clean setup jupyter

#################################################################################
# GLOBALS                                                                       #
#################################################################################
CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV)

#################################################################################
# COMMANDS                                                                      #
#################################################################################
setup: $(CONDA_ENV)
	git config core.hooksPath .githooks

$(CONDA_ENV): environment.yml
	mamba env create --force --prefix $(CONDA_ENV) --file environment.yml

jupyter:
	$(CONDA_ACTIVATE); export PYTHONPATH=`pwd`:$${PYTHONPATH}; jupyter lab --notebook-dir=notebooks/

clean:
	rm -rf env/
