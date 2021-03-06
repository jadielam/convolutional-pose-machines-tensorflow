# To Build:
# docker build -t pose .

# To run:
# nvidia-docker run -p 8888:8888 -v /home/ubuntu/src/notebooks:/src/notebooks -t pose

FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

# Installing dependencies for python packages
RUN apt-get update -y && apt-get -y install \
    gcc \
    libgtk2.0-dev \
    curl \
    vim

# Installing conda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Use the environment.yaml to create the conda environment
ADD environment.yaml .
RUN conda env create -f environment.yaml

# Creating src folder
RUN mkdir src
RUN mkdir src/notebooks
ADD code/ src/convolutional-pose-machines-tensorflow
RUN [ "/bin/bash", "-c", "source activate pose" ]

WORKDIR src/notebooks
ENV PYTHONPATH=:src/convolutional-pose-machines-tensorflow
EXPOSE 8888

# Running jupyter notebook as entry point
ENTRYPOINT ["/bin/bash", "-c", "source activate pose && jupyter notebook --ip='*' --allow-root --no-browser --port=8888"]
#ENTRYPOINT ["/bin/bash"]
