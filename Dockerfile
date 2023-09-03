FROM ubuntu:22.04
#FROM nvidia/cuda:11.2-runtime-ubuntu20.04

RUN apt-get update && \
    apt-get install -y python3-pip

RUN pip3 install numpy scipy pandas
RUN pip3 install scikit-learn
RUN pip3 install tensorflow tensorflow_datasets
RUN pip3 install matplotlib seaborn

RUN apt-get install -y git && \
    apt-get install -y libsndfile1
RUN pip3 install soundfile
RUN pip3 install librosa
RUN git clone https://github.com/kan-bayashi/ParallelWaveGAN.git
RUN pip3 install parallel_wavegan