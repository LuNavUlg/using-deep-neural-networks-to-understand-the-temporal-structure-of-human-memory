# Using deep neural networks to understand the temporal structure of human memory

This repository contains the code and materials related to my master thesis titled "Understanding the Temporal Structure of Human Memory using Deep Neural Networks". This thesis was submitted in partial fulfillment of the requirements for the degree of Engineering and Computer Science at the University of Liège, Belgium, in June 2023. The thesis was supervised by Prof. Gilles Louppe and Prof. Arnaud D'Argembeau.

## Overview

In this thesis, I investigate the temporal dynamics of human memory and explore the use of deep neural networks to model and predict recall times of episodic memories. The research focuses on analyzing the performance of different neural network architectures and identifying critical layers that contribute to accurate recall time predictions.

## Repository Structure

- `expx/`: These directories contain the output of the experiments performed in the thesis. Each directory contains the log files, the results and evaluation metrics obtained from running the models on the dataset, as well as the parameters used for training the models. Each directory is named after the experiment it contains, according to the table below.
- `results/`: This directory contains the sense of time and change detection results obtained from running the models on the dataset.
- `README.md`: This file provides an overview of the repository and its contents (you are reading it right now).
- `baseline.py`: This file contains the code for the baseline models (dummy baseline and real-duration baseline) used in the thesis.
- `calibration.py`: This file contains the code for the calibration of the parameters of the accumulation mechanism used in the thesis.
- `DRTANet_script.py`: This file contains the code for the experiments using the DRTANet model. The DR-TANet model is based on the TANet model proposed by [Chen, Shuo, Kailun Yang, and Rainer Stiefelhagen (2021)](https://github.com/Herrccc/DR-TANet). This file makes use of the `changes.p`, `DRTANet_accs_64.p` and `DRTANet_history_64.p` files.
- `process.py`, `regression.py`, `script.py`, `utils.py`: These files contain the code for the experiments described in the thesis and in the table below.

The rest of the files are either generated by the code (and left in this repository for convenience of future users who might want to reproduce the experiments) or log files generated by the code. The `.sbatch` files are used to run the code on the Alan cluster of the University of Liège (also left in this repository for convenience of future users who might want to reproduce the experiments).

## Usage

### Table of experiments:
| Name  | Model           | Vanilla            | Lasso              | Naive              |
| ----- | --------------- | ------------------ | ------------------ | ------------------ |
| exp1  | AlexNet         | :white_check_mark: | :x:                | :x:                |
| exp2  | AlexNet         | :x:                | :white_check_mark: | :x:                |
| exp3  | AlexNet         | :x:                | :x:                | :white_check_mark: |
| exp4  | ResNet-18       | :x:                | :white_check_mark: | :x:                |
| exp5  | ResNet-18       | :x:                | :x:                | :white_check_mark: |
| exp6  | EfficientNet-B0 | :x:                | :white_check_mark: | :x:                |
| exp7  | EfficientNet-B0 | :x:                | :x:                | :white_check_mark: |
| exp8  | EfficientNet-B4 | :x:                | :white_check_mark: | :x:                |
| exp9  | EfficientNet-B4 | :x:                | :x:                | :white_check_mark: |
| exp10 | EfficientNet-B7 | :x:                | :white_check_mark: | :x:                |
| exp11 | EfficientNet-B7 | :x:                | :x:                | :white_check_mark: |

The experiment `exp0` computes two different baselines to compare the other models to: a dummy baseline, and a baseline based on the real duration of the videos.
The experiment `exp0bis` computes two different versions of the pipeline using DR-TANet: one with the final values of the accumulators computed using DR-TANet, and one with the complete history of the accumulators computed using DR-TANet.

To use the code and reproduce the experiments, follow these steps:

1. Clone the repository: `git clone https://github.com/LuNavUlg/using-deep-neural-networks-temporal-structure-human-memory.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the videos (not linked here for privacy and storage limitations reasons) and place them in a `Videos/` directory in the root of the repository.
4. Run the `run.sbatch` file to run the code on the Alan cluster of the University of Liège. The script will run the code for all the experiments described in the table above. The results will be stored in the `expx/` directories. The `run.sbatch` file can be modified to run only a subset of the experiments. The `run.sbatch` file also makes use of arguments that can be modified to change the parameters of the experiments. 
Run the following command: 
   ```
   sbatch run.sbatch $1 $2 $3
   ```
   The arguments are described in the list below:
   - `$1 (int)`: The number of videos to use for the experiments (64 for all the videos of the dataset).
   - `$2 (bool)`: Compute accumulators (True) or re-use the ones already computed (False).
   - `$3 (str)`: Calibration method to use for the accumulation mechanism (``fixed``, ``stats``).
