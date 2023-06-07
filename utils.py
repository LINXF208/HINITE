import numpy as np 
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.utils import shuffle
import pandas as pd
import math
import os
import evaluation
from tensorflow import keras
import matplotlib.pyplot as plt
import HINITE
import scipy.io as scio





def COMP_HSIC(X,t,s_x=1,s_y=1):

    """ Computes the HSIC(X,t)"""
    K = GaussianKernelMatrix(X,s_x)
    L = GaussianKernelMatrix(t,s_y)
    m = X.shape[0]
    H = tf.raw_ops.MatrixDiag(diagonal = tf.ones(shape=[m,])) - 1/m
    LH = tf.matmul(L,H)
    HLH = tf.matmul(H,LH)
    KHLH = tf.matmul(K,HLH)
    #print("check hsic",K,KHLH,tf.linalg.trace(KHLH))
    HSIC = tf.linalg.trace(KHLH)/((m-1)**2)
    print("check hsic",tf.linalg.trace(KHLH))
    #print(KHLH)
    return HSIC

def GaussianKernelMatrix(x,sigma = 1):

    """ Computes the Gaussian Kernel Matrix"""
    pairwise_distances = pdist2sq(x,x) # Computes the squared Euclidean distance
    return tf.exp(-pairwise_distances/2*sigma)

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*tf.matmul(X,tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X),1,keepdims=True)
    ny = tf.reduce_sum(tf.square(Y),1,keepdims=True)
    D = (C + tf.transpose(ny)) + nx
    return D    

def divide_TC(concated_data,input_t):
    #temp = tf.concat([hidden,GNN,weighted_G],1)
    #print("input_t",input_t)
    i0 = tf.cast((tf.where(input_t < 1)[:,0]),tf.int32)
    i1 = tf.cast((tf.where(input_t > 0)[:,0]),tf.int32)
    #mask = np.logical_and(np.array(input_t)[:,-1] == 1,1)
    #print("concated_data",concated_data)
    #print("mask",mask)
    group_T = tf.gather(concated_data,i1)
    #print("group_T",group_T)
    group_C = tf.gather(concated_data,i0)


    return tf.constant(group_T),tf.constant(group_C),i0,i1

def split_train_val_test(data,train_ratio,val_ratio,test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    train_set_size = int(len(data) * train_ratio)
    val_set_size = int(len(data) * val_ratio)
    train_indices = shuffled_indices[:train_set_size]
    val_indices = shuffled_indices[train_set_size:train_set_size+val_set_size]
    test_indices = shuffled_indices[train_set_size+val_set_size:]
    return train_indices,val_indices,test_indices

def train(Model_name,input_data,y,train_idx,val_idx,config_hyperparameters, max_iterations, lr_rate,lr_weigh_decay,flag_early_stop=False,activation = tf.nn.elu,true_ite = [],cur_adj=[]):
    cur_all_features = input_data[:,:-1]
    cur_init_A = cur_adj
    cur_model = Model_name(config_hyperparameters,activation=activation,init_adj = cur_init_A) 
    count = 0
    losslist_CV = []
    sum_loss = 0
    sum_CV_loss = 0
    losslist = []
    for i in range(max_iterations):
        #print("iter",i)
        loss = cur_model.CV_y(tf.cast(input_data,tf.float32),tf.cast(y,tf.float32),train_idx)
        total_loss = cur_model.network_learn(tf.cast(input_data,tf.float32),tf.cast(y,tf.float32),train_idx,learning_rate=lr_rate,learning_w_decay =lr_weigh_decay)
        CV_loss = cur_model.CV_y(tf.cast(input_data,tf.float32),  tf.cast(y,tf.float32),val_idx)
        sum_loss += loss
        sum_CV_loss += CV_loss
        if (i+1) % 20 == 0:
            if len(losslist_CV) > 0 and sum_CV_loss/20 >= losslist_CV[-1]:
                count += 1
            else:
                count = 0
            if flag_early_stop:
                if i > 400 and count >= 1:
                    break
            losslist.append(sum_loss/20)
            losslist_CV.append(sum_CV_loss/20)
            sum_loss = 0
            sum_CV_loss = 0
        if len(true_ite) > 0:
            c_val_T, c_val_C = cur_model.pre_no_interf(tf.cast(input_data,tf.float32),val_idx)
            c_val_ite = c_val_T-c_val_C
            #print("cur pehe", np.mean((c_val_ite - true_ite)**2))
    plt.plot(range(len(losslist)),losslist,label="Train_loss")
    plt.plot(range(len(losslist_CV)),losslist_CV,label = "Validation_loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.show()
    plt.close()

    return cur_model

 
def save_mymodel(save_path,save_name,need_save_model):
	cur_path = save_path + '/' + save_name
	need_save_model.save_weights(cur_path )
	print("Already saved the model's weights in file" + cur_path  )

def load_mymodel(load_path,load_name,need_load_model,config_hyperparameters,activation,init_A):
	cur_model = need_load_model(config_hyperparameters,activation,init_A)
	cur_path = load_path + '/' + load_name
	cur_model.load_weights(cur_path)
	print("load model")
	return cur_model






def find_hyperparameter(set_configs,data_name,Model_name,activation,start_split_i,end_split_i):
  

    data = load_data(data_name)
    train_idx,val_idx,test_idx = split_train_val_test(data[0][0],0.7,0.15,0.15)
    for i in range(start_split_i,end_split_i):
        cur_all_input, yf,mu1,mu0,adjs= data_preparation(data_name,i,data)
        true_ite = mu1 - mu0
        val_true_ite = true_ite[val_idx]
        print("val true ite",val_true_ite)
        val_true_ate = np.mean(true_ite[val_idx])
        test_true_ite = true_ite[test_idx]
        print("test ite",val_true_ite)
        test_true_ate = np.mean(true_ite[test_idx])
        for j in range(len(set_configs)):
            cur_val_results = []
            config = set_configs[j]
            cur_model = train(Model_name,tf.cast(cur_all_input,tf.float32),tf.cast(yf,tf.float32),train_idx,val_idx,config, config["iterations"], config["lr_rate"],config["lr_weigh_decay"],config["flag_early_stop"],activation = activation,true_ite= val_true_ite,cur_adj=adjs)
            cur_save_model_name = "model"
            cur_save_path = './save_Models_HINITE/Model_Youtube1.01.0'+ str(Model_name)[8:-2]  +  '_'+ "divide_"+str(config['divide'])+"_" "split_" + str(i)
            if config['rep_alpha'] > 0:
                cur_save_path += 'rep_alpha' + str(config['rep_alpha'])
            
            os.makedirs(cur_save_path,exist_ok=True)
            save_mymodel(cur_save_path,cur_save_model_name,cur_model)


def save_results(save_result,save_name):
	np.save(save_name,save_result) 
	print("saved all results ")

def load_data(data_name):

    data = []

    
    if data_name == 'Youtube':
        all_x = []
        all_t = []
        all_yf = []
        all_m1 = []
        all_m0 = []
        all_A = []
        for i in range(10):
            cur_name = "./data/Youtube/1.0k0_1.0k1_1.0ad_Youtube_"
            
            cur_name_x = cur_name + "x_" + str(i)+"_5000.npy"
            cur_name_t = cur_name + "T_" + str(i)+"_5000.npy"
            cur_name_yf = cur_name + "yf_" + str(i)+"_5000.npy"
            cur_name_m1 = cur_name + "y1_spe_" + str(i)+"_5000.npy"
            cur_name_m0 = cur_name + "y0_spe_" + str(i)+"_5000.npy"
            cur_name_A = cur_name + "adjs_" + str(i)+"_5000.npy"
            
            cur_x = np.load(cur_name_x)
            cur_t = np.load(cur_name_t)
            cur_yf = np.load(cur_name_yf)
            cur_m1 = np.load(cur_name_m1)
            cur_m0 = np.load(cur_name_m0)
            
            cur_A = np.load(cur_name_A)
            all_x.append(cur_x)
            all_t.append(cur_t)
            all_yf.append(cur_yf)
            all_m1.append(cur_m1)
            all_m0.append(cur_m0)
            all_A.append(cur_A)
        data = []
        data.append(all_x)
        data.append(all_t)
        data.append(all_yf)

        data.append(all_m1)
        data.append(all_m0)
        data.append(all_A)
        return data



def data_preparation(data_name,split_idx,data):

   
   
    if data_name == 'Youtube':
        all_x = data[0]
        all_t = data[1]
        all_yf = data[2]
        all_m1 = data[3]
        all_m0 = data[4]
        all_A = data[5]

        cur_x = all_x[split_idx]
        cur_t = all_t[split_idx]
        cur_yf = all_yf[split_idx]
        cur_m1 = all_m1[split_idx]
        cur_m0 = all_m0[split_idx]
        cur_A = all_A[split_idx]



        cur_t = cur_t.reshape(len(cur_t),1)
        cur_yf = cur_yf.reshape(len(cur_yf),1)
        cur_m1 = cur_m1.reshape(len(cur_m1),1)
        cur_m0 = cur_m0.reshape(len(cur_m0),1)

        cur_all_input = np.concatenate([cur_x,cur_t],1)

        return cur_all_input,cur_yf,cur_m1,cur_m0,cur_A


def config_pare_HINITE(iterations,lr_rate,lr_weigh_decay,flag_early_stop,use_batch,rep_alpha,flag_norm_gnn,flag_norm_rep,
                           out_dropout,GNN_dropout,rep_dropout,inp_dropout,rep_hidden_layer,rep_hidden_shape,
                           GNN_hidden_layer,GNN_hidden_shape,head_num_att,out_T_layer,out_C_layer,
                           out_hidden_shape,activation,GNN_alpha,divide,hete_att_size):

        all_configs = []

        for i in range(len(rep_alpha)):
            cur_activation = activation

            config = {}
            config["iterations"] = iterations
            config["lr_rate"] = lr_rate
            config["lr_weigh_decay"] = lr_weigh_decay
            config["flag_early_stop"] = flag_early_stop
            config['rep_alpha'] = rep_alpha[i]
            config['flag_norm_gnn'] = flag_norm_gnn
            config['flag_norm_rep'] = flag_norm_rep
            config['out_dropout'] = out_dropout
            config['GNN_dropout'] = GNN_dropout
            config['rep_dropout'] = rep_dropout
            config['inp_dropout'] = inp_dropout
            config['use_batch'] = use_batch
            config['rep_hidden_layer'] = rep_hidden_layer
            config['rep_hidden_shape'] = rep_hidden_shape
            config['GNN_hidden_layer'] = GNN_hidden_layer
            config['GNN_hidden_shape'] = GNN_hidden_shape
            config['head_num_att'] = head_num_att
            config['out_T_layer'] = out_T_layer
            config['out_C_layer'] = out_C_layer
            config['out_hidden_shape'] = out_hidden_shape
            config['GNN_alpha'] = GNN_alpha[0]
            config['divide'] = divide
            config['hete_att_size'] = hete_att_size

            all_configs.append(config)

        return all_configs,cur_activation




 
        