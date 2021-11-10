FROM ubuntu:latest

# install anaconda
RUN apt-get update
RUN apt-get install -y wget && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# setup conda virtual environment
COPY environment.yml .
RUN conda install mamba -n base -c conda-forge
RUN mamba env create --name {{cookiecutter.project_name}} -f environment.yml
RUN R -e "IRkernel::installspec()"

RUN echo "conda activate {{cookiecutter.project_name}}" >> ~/.bashrc
ENV PATH /opt/conda/envs/{{cookiecutter.project_name}}/bin:$PATH
ENV CONDA_DEFAULT_ENV ${{cookiecutter.project_name}}

# set up project
RUN mkdir /{{cookiecutter.project_name}}
WORKDIR {{cookiecutter.project_name}}
EXPOSE 8888

