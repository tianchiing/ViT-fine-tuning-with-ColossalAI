## Overview

This is a repository cloned from https://github.com/hpcaitech/ColossalAI/tree/main/examples as required by a course assignment.

The model used in this experiment is Vision Transformer (ViT), a visual model based on the architecture of a transformer for computer vision tasks. It was published in paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

In this example originally from Colossal-AI, the pretrained weights of ViT loaded from HuggingFace was used.

The dataset used in this example's demo is [Beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans). This is a set of images of beans leaf, labelled with ['angular_leaf_spot', 'bean_rust', 'healthy'].

This example supports plugins including TorchDDPPlugin (DDP), LowLevelZeroPlugin (Zero1/Zero2), GeminiPlugin (Gemini) and HybridParallelPlugin (any combination of tensor/pipeline/data parallel).


## Before Running

Make sure the dependencies are configured correctly.

You need to have cuda version 11.6 and corresponding pytorch to run the example.

If your cuda version is not 11.6, or if you get relevant error messages. Check your cuda version 
```bash
lspci | grep -i nvidia
```

If you have previous installation remove it first, run the following command to install the expected cuda driver 11.3
```bash
sudo apt-get purge nvidia*
sudo apt remove nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

## system update
sudo apt-get update
sudo apt-get upgrade

# install other import packages
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# first get the PPA repository driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# install nvidia driver with dependencies
sudo apt install libnvidia-common-470
sudo apt install libnvidia-gl-470
sudo apt install nvidia-driver-470


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.6.0/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.0-510.39.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
sudo apt-get update

sudo add-apt-repository ppa:cloudhan/liburcu6
sudo apt update
sudo apt install liburcu6


sudo apt install cuda-11-6

# setup your paths
echo 'export PATH=/usr/local/cuda-11.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# Finally, to verify the installation, check
nvidia-smi
nvcc -V
```

With Cuda 11.6, run the following command to install pytorch of specific version, and flash-attn dependencies.

```bash
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116

pip install flash-attn --no-build-isolation
```

You also need to make sure that GCC versions is NOT LATER than 10. Downgrade the GCC version if required.

```bash
gcc --version

sudo apt-get install gcc-9 g++-9 -y
sudo ln -s /usr/bin/gcc-9 /usr/bin/gcc
sudo ln -s /usr/bin/g++-9 /usr/bin/g++
sudo ln -s /usr/bin/gcc-9 /usr/bin/cc
sudo ln -s /usr/bin/g++-9 /usr/bin/c++
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 1
sudo update-alternatives --config gcc

```


## Run Demo

By running the following script:
```bash
bash run_demo.sh
```


## Run Benchmark

You can run benchmark for ViT model by running the following script:
```bash
bash run_benchmark.sh
```
The script will test performance (throughput & peak memory usage) for each combination of hyperparameters.


## Experiment Result

Experiment on A10:

run_demo.sh
Epoch | #1 | #2 | #3 
--- | --- | --- | --- 
it/s | 5.86 | 7.12 | 7.14 
Average Loss | 0.0969 | 0.0716 | 0.0608
Accuracy | 0.9766 | 0.9609 | 0.9688

<br /><br />

run_benchmark.sh

batch size per gpu: 8
plugin | torch_ddp | torch_ddp_fp16 | low_level_zero | gemini | hybrid_parallel
--- | --- | --- | --- | --- | --- 
throughput | 33.7902 | 50.0516 | 46.4627 | 34.8441 | 41.4561
maximum memory usage per gpu | 1.80 GB | 1.91 GB | 1.65 GB | 663.17 MB | 1.73 GB
it/s | 4.22 | 6.26 | 5.81 | 4.36 | 5.18

batch size per gpu: 32
plugin | torch_ddp | torch_ddp_fp16 | low_level_zero | gemini | hybrid_parallel
--- | --- | --- | --- | --- | --- 
throughput | 55.2553 | 115.7007 | 122.8285 | 115.3535 | 121.0082
maximum memory usage per gpu | 2.34 GB | 2.25 GB | 1.66 GB | 663.17 MB | 1.92 GB
it/s | 1.73 | 3.62 | 3.84 | 3.61 | 3.78
