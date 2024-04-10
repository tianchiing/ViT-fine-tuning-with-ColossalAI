## Overview

This is a repository cloned from https://github.com/hpcaitech/ColossalAI/tree/main/examples as required by a course assignment.

The model used in this experiment is Vision Transformer (ViT), a visual model based on the architecture of a transformer for computer vision tasks. It was published in paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

In this example originally from Colossal-AI, the pretrained weights of ViT loaded from HuggingFace was used.

The dataset used in this example's demo is [Beans](https://huggingface.co/datasets/AI-Lab-Makerere/beans). This is a set of images of beans leaf, labelled with diseased or healthy leaf.

This example supports plugins including TorchDDPPlugin (DDP), LowLevelZeroPlugin (Zero1/Zero2), GeminiPlugin (Gemini) and HybridParallelPlugin (any combination of tensor/pipeline/data parallel).


## Before Running

Make sure the dependencies are configured correctly.

You need to have cuda version 11.3 and corresponding pytorch to run the example.

```bash
pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

If your cuda version is not 11.3, or if you get relevant error messages. Check your cuda version 
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
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

 # installing CUDA-11.3
sudo apt install cuda-11-3

# setup your paths
echo 'export PATH=/usr/local/cuda-11.3/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# install cuDNN v11.3
# First register here: https://developer.nvidia.com/developer-program/signup

CUDNN_TAR_FILE="cudnn-11.3-linux-x64-v8.2.1.32.tgz"
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/cudnn-11.3-linux-x64-v8.2.1.32.tgz
tar -xzvf ${CUDNN_TAR_FILE}

# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.3/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.3/lib64/
sudo chmod a+r /usr/local/cuda-11.3/lib64/libcudnn*

# Finally, to verify the installation, check
nvidia-smi
nvcc -V
```

You also need to make sure that GCC versions is NOT LATER than 10. Downgrade the GCC version if required.

```bash
gcc --version

sudo apt remove gcc
sudo apt-get install gcc-9 g++-9 -y
sudo ln -s /usr/bin/gcc-9 /usr/bin/gcc
sudo ln -s /usr/bin/g++-9 /usr/bin/g++
sudo ln -s /usr/bin/gcc-9 /usr/bin/cc
sudo ln -s /usr/bin/g++-9 /usr/bin/c++
```


## Run Demo

By running the following script:
```bash
bash run_demo.sh
```
You will finetune a a [ViT-base](https://huggingface.co/google/vit-base-patch16-224) model on this [dataset](https://huggingface.co/datasets/beans), with more than 8000 images of bean leaves. This dataset is for image classification task and there are 3 labels: ['angular_leaf_spot', 'bean_rust', 'healthy'].

The script can be modified if you want to try another set of hyperparameters or change to another ViT model with different size.

The demo code refers to this [blog](https://huggingface.co/blog/fine-tune-vit).



## Run Benchmark

You can run benchmark for ViT model by running the following script:
```bash
bash run_benchmark.sh
```
The script will test performance (throughput & peak memory usage) for each combination of hyperparameters. You can also play with this script to configure your own set of hyperparameters for testing.


## Experiment Result

