# PPO相关9：开源readme

# Overview

This repository is the official implementation of the article **"Efficient navigation in vortical flows based on reinforcement learning and flow field prediction"**. The goal of this work is to enhance reinforcement learning (RL) navigation in complex flow fields using the **Look-Ahead State Space (LASS)** method. LASS allows RL agents to effectively utilize flow field prediction information, improving navigation performance in environments with partially observable and dynamic flow fields. The core innovation lies in enabling RL agents to navigate in situations where the agent's velocity is slower than the flow field's, thereby requiring the agent to rely on flow field features for effective navigation. The article demonstrates that a simple yet effective strategy, like LASS, can significantly improve RL's efficiency.

The repository includes the implementation of RL algorithms, flow field prediction models, and related tools, allowing researchers to replicate the experiments and further explore RL-based navigation tasks in fluid dynamics.

[DG.mp4](video/DG_65pxiRDaad.mp4)

[CF.mp4](video/CF_2TfV1hFFnb.mp4)



| Environment    | SuccessRate-BaseSS  | SuccessRate-TrueLASS  | SuccessRate-PredLASS  |
| -------------- | ------------------- | --------------------- | --------------------- |
| Double-Gyre    | 69.23%              | 95.45%                | 94.90%                |
| Cylinder-Flow  | 64.75%              | 98.91%                | 98.29%                |

# Requirements

The code has been tested with Python 3.9 and PyTorch 1.13 on Win10, Ubuntu 20.04, and Ubuntu 22.04 machines. We recommend using a Conda environment for setup.

Note that due to the nature of reinforcement learning, which requires continuous information exchange between the RL agent and the environment (typically running on the CPU), it is recommended to run all training and testing on the CPU rather than GPU, except for the training of the flow field prediction network.

To run the code on a Windows machine with Conda installed, follow the instructions below:

```bash
# For the CPU version.
conda create -n LASS python=3.9
conda install pytorch=1.13.1 -c pytorch -c nvidia -y
conda install gymnasium termcolor colorma scipy matplotlib scikit-learn pandas -y

```


```bash
# For the GPU version. Adjust the PyTorch version based on your system specifications.
conda create -n LASS python=3.9
conda install pytorch=1.13.1=py3.9_cuda11.7_cudnn8_0 -c pytorch -c nvidia -y
conda install gymnasium termcolor colorma scipy matplotlib scikit-learn pandas -y
```


Some files are too large to upload to this repo. Please download mannually from one of the links below:

Download from Google Drive: [https://drive.google.com/drive/folders/16L7Sb2jvPnujp3NuRVxnmz4R7JcUrD8Z?usp=sharing](https://drive.google.com/drive/folders/16L7Sb2jvPnujp3NuRVxnmz4R7JcUrD8Z?usp=sharing "https://drive.google.com/drive/folders/16L7Sb2jvPnujp3NuRVxnmz4R7JcUrD8Z?usp=sharing")

Download from BaiduNetDisk(百度网盘): [https://pan.baidu.com/s/1Hv308yQta1hhITsXnl8KnA?pwd=vsqt](https://pan.baidu.com/s/1Hv308yQta1hhITsXnl8KnA?pwd=vsqt "https://pan.baidu.com/s/1Hv308yQta1hhITsXnl8KnA?pwd=vsqt")



# File Structure

The repository is organized as follows:

- **`my_environment.py`**: Defines the simulation environments for **Double-Gyre** and **Cylinder-Flow**, which are used for reinforcement learning-based navigation experiments.
- **`env_obs_strategy.py`**: Specifies the state-space (SS) used by the RL agent during navigation.
- **`my_PPO.py`**: Implements the **Proximal Policy Optimization (PPO)** algorithm, used for the RL experiments in this study.
- **`field_predict_CF_Re400.py`**: Contains the training and usage code for the **Cylinder-Flow** flow field prediction network.
- **`field_predict_double_gyre.py`**: Contains the training and usage code for the **Double-Gyre** flow field prediction network.
- **`test.py`**: Used to test the trained RL model.
- **`train_RL.py`**: Used to train the RL agent.
- **`flow_field/`**: This folder contains the grid data files (.mat) for the two flow fields used in the experiments. DOWNLOAD MANNUALLY FROM SUPPLEMENTARY LINK!
- **`pretrained_models/`**: This folder includes pre-trained models for both the PPO network and flow field prediction networks. DOWNLOAD MANNUALLY FROM SUPPLEMENTARY LINK!
- **`field_prediction_training_data_gen/`**: This folder includes training data generation algorithms for field prediction network.&#x20;

# Using Pretrained Models

To validate the results in this article, simply modify the configuration parameters in **`test.py`** and run the code. All key settings are specified at the beginning of the **`test.py`** file.

# Further Training

**To train the RL agent**, modify the configuration parameters in`train_RL.py`and run the code with the specified model ID and number of training times. For example, to train 5 different agents start with ID=1, use the following command:

```bash
python train_RL.py 1 5
```


This will train 5 RL agents with IDs 1 to 5, and you can select the best-performing one afterward. If you wish to train another batch, ensure that you avoid reusing existing IDs. For instance, start the ID from 6 as follows:

```bash
python train_RL.py 6 5
```


**To train the flow field prediction network**, you first need to generate training data. The training data can be generated using Python files located in the`field_prediction_training_data_gen/`folder. The algorithm can simulate the agent's random movement within a specified environment, recording itstrajectory-flow field dataas training data. You can also manually control the agent's movements to cover areas that random movements might not reach. For detailed information, please refer to the Python files. Once the training data is generated, you can call the corresponding`field_predict_XXX.py`file to train the network.
