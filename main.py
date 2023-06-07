import utils
import GLITE
import ourlayers
import evaluation
import numpy as np 
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.utils import shuffle
import pandas as pd
import math
import statsmodels.api as sm
from tensorflow import keras
from sklearn.model_selection import train_test_split
import seaborn as sns
import HINITE



def main_YB_HINITE():

    configs,activation = utils.config_pare_HINITE(iterations=2000,lr_rate=0.001,lr_weigh_decay=0.001,flag_early_stop=True,use_batch=512,
    rep_alpha=[0.01,0.1,0.5,1.0,1.5],flag_norm_gnn=False,flag_norm_rep=False,out_dropout=0.2,GNN_dropout=0.2,rep_dropout=0.2,inp_dropout=0.0,
    rep_hidden_layer=3,rep_hidden_shape=[128,64,64],GNN_hidden_layer=3,GNN_hidden_shape=[64,64,32], divide = True,
    head_num_att=1,out_T_layer=3,out_C_layer=3,out_hidden_shape= [128,64,32],activation = tf.nn.relu,GNN_alpha = [0.0],hete_att_size = [128,128,64]
    )

    utils.find_hyperparameter(set_configs=configs,data_name='Youtube',
                              Model_name=HINITE.HINITEModel,activation=activation,start_split_i=0,end_split_i = 10)
if __name__ == '__main__':
    main_YB_HINITE()



