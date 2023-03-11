FROM ubuntu:20.04
#FROM nvidia/cuda:11.2-runtime-ubuntu20.04

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install numpy scipy tensorflow
RUN pip3 install pandas tensorflow_datasets
RUN pip3 install matplotlib seaborn
RUN pip3 install scikit-learn