###############################
# GLOBALS
CONDA_ENV = ./env/
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)"; conda activate $(CONDA_ENV); export PYTHONPATH=`pwd`:$${PYTHONPATH}

.PHONY: $(CONDA_ENV) clean setup jupyter

###############################
# COMMANDS
setup: $(CONDA_ENV)
	$(CONDA_ACTIVATE) && R -e "IRkernel::installspec()"
	pre-commit install

docker_build: Dockerfile
	docker build -t {{cookiecutter.project_name}} .

$(CONDA_ENV): environment.yml
	mamba env create --force --prefix $(CONDA_ENV) --file environment.yml

jupyter:
	$(CONDA_ACTIVATE); export PYTHONPATH=`pwd`:$${PYTHONPATH}; jupyter lab --notebook-dir=notebooks/

clean:
	rm -rf env/
