# HINITE
![HINITE](https://github.com/LINXF208/HINITE/edit/main/A4.png)
![$\psi$](https://github.com/LINXF208/HINITE/edit/main/A5.png)
![HIA](https://github.com/LINXF208/HINITE/edit/main/A6.png)
## Overview
- data/ contains the datasets that used in our experiments.
- save_Models_HINITE will be used to save the trained models.
- results will be used to save the results.
- ourlayers.py contains layers of the HINITE.
- HINITE.py contains implementation of the HINITE.
- evaluation.py contains calucation of the ATE and ITE.
- main.py It is used to execute a full training run on the Youtube dataset.
- Evaluate_for_HINITE.ipynb Evaluate the HINITE on the Youtube dataset.
## Requirements
- python==3.8.13
- numpy==1.23.4
- jupyter==1.0.0
- notebook==6.5.1
- pandas==1.5.0
- matplotlib==3.6.1
- scipy==1.9.2
- tensorflow-gpu==2.4.1
- scikit-learm==1.1.2
- seaborn==0.12.0 

Our experiments are performed by RTX A5000 GPU.  In addition, you need to install the cuDNN8.0 and CUDA11.0.
## Train the HINITE
Here, we give an example of training the HINITE using the Youtube dataset.

1. First, you need to decompress the dataset in the "data" folder. 
2. Then, you can train the HINITE by runing main.py. For example, CUDA_VISIBLE_DEVICES=1 python main.py
3. Finally, you can open the "Evaluate_for_HINITE.ipynb", and run every cell to evaluate the HINITE. 



