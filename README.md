# HINITE

## Overview
- data/: It contains the datasets that are used in our experiments.
- save_Models_HINITE: It is used to save the trained models.
- results: It is used to save the results.
- ourlayers.py: It contains layers of the HINITE.
- HINITE.py: It contains the implementation of the HINITE.
- evaluation.py: It contains the calculation of the ATE and ITE.
- main.py: It is used to execute a full training run on the Youtube dataset.
- Evaluate_for_HINITE.ipynb: It is used to evaluate the HINITE on the Youtube dataset.
## Requirements
- python==3.8.13
- numpy==1.23.4
- jupyter==1.0.0
- notebook==6.5.1
- pandas==1.5.0
- matplotlib==3.6.1
- scipy==1.9.2
- TensorFlow-gpu==2.4.1
- scikit-learm==1.1.2
- seaborn==0.12.0

Our experiments are performed by RTX A5000 GPU.  In addition, you need to install cuDNN8.0 and CUDA11.0.
## Datasets
The datasets with simulated outcomes can be downloaded at https://www.dropbox.com/sh/6e811ndfc4sdfy1/AABynXpVLl4uaj48YiTlo7kWa?dl=0.
## Train the HINITE
Here, we give an example of training the HINITE using the Youtube dataset.

1. First, you need to download and decompress datasets and put them into the "data" folder. 
2. Then, you can train the HINITE by running main.py. For example, CUDA_VISIBLE_DEVICES=1 python main.py
3. Finally, you can open the "Evaluate_for_HINITE.ipynb", and run every cell to evaluate the HINITE. 



