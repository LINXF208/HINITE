import math
import os

import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import HINITE
import evaluation


def comp_hsic(X, t, s_x=1, s_y=1):
    """ Compute the HSIC. 
    Args:
        X (tf.Tensor): Representation matrix.
        t (tf.Tensor): Treatment assignment vector (binary: 0 or 1).
        p (float): Probability of treatment.
    Returns:
        tf.Tensor: computed HSIC.
    """
    K = GaussianKernelMatrix(X, s_x)
    L = GaussianKernelMatrix(t, s_y)
    m = X.shape[0]

    H = tf.raw_ops.MatrixDiag(diagonal=tf.ones(shape=[m, ])) - 1 / m
    LH = tf.matmul(L, H)
    HLH = tf.matmul(H, LH)
    KHLH = tf.matmul(K, HLH)
    HSIC = tf.linalg.trace(KHLH) / ((m - 1) ** 2)

    return HSIC


def GaussianKernelMatrix(x, sigma=1):
    """ Computes the Gaussian Kernel Matrix"""
    pairwise_distances = pdist2sq(x, x) # Computes the squared Euclidean distance
    return tf.exp(-pairwise_distances / 2 * sigma)


def pdist2sq(X, Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keepdims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keepdims=True)
    D = (C + tf.transpose(ny)) + nx

    return D    


def divide_t_c(concated_data, input_t):
    """Divide units into Treated and Control groups.

    Args:
        concated_data (tf.Tensor): The dataset containing all units.
        input_t (tf.Tensor): Binary tensor (0: Control, 1: Treated).

    Returns:
        tuple: (group_T, group_C, i0, i1)
            - group_T (tf.Tensor): Treated group data.
            - group_C (tf.Tensor): Control group data.
            - i0 (tf.Tensor): Indices of the Control group.
            - i1 (tf.Tensor): Indices of the Treated group.
    """
    i0 = tf.cast((tf.where(input_t < 1)[:, 0]), tf.int32)
    i1 = tf.cast((tf.where(input_t > 0)[:, 0]), tf.int32)

    group_T = tf.gather(concated_data, i1)
    group_C = tf.gather(concated_data, i0)

    return group_T, group_C, i0, i1


def split_train_val_test(data, train_ratio, val_ratio, test_ratio,seed=42):
    """
    Split data indices into training, validation, and test sets.

    Args:
        data (array-like): The dataset to split.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.
        test_ratio (float): Proportion of the dataset to use for testing.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    if len(data) == 0:
        raise ValueError("Data cannot be empty.")

    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))

    train_set_size = int(len(data) * train_ratio)
    val_set_size = int(len(data) * val_ratio)

    train_indices = shuffled_indices[:train_set_size]
    val_indices = shuffled_indices[train_set_size:train_set_size + val_set_size]
    test_indices = shuffled_indices[train_set_size + val_set_size:]
    
    return train_indices, val_indices, test_indices


def train(
        Model_name,
        input_data,
        y,
        train_idx,
        val_idx,
        config_hyperparameters, 
        max_iterations, 
        lr_rate,
        lr_weigh_decay,
        flag_early_stop=False,
        activation=tf.nn.elu,
        true_ite=[],
        cur_adj=[]
    ):
    cur_all_features = input_data[:, :-1]
    cur_model = Model_name(config_hyperparameters, activation=activation, init_adj=cur_adj) 

    count = 0
    sum_loss = 0
    sum_CV_loss = 0
    losslist = []
    losslist_CV = []

    for i in range(max_iterations):
        print("iter", i)
        loss = cur_model.val_y(tf.cast(input_data, tf.float32), tf.cast(y, tf.float32), train_idx)
        total_loss = cur_model.network_learn(
            tf.cast(input_data, tf.float32), 
            tf.cast(y, tf.float32), 
            train_idx
        )
        CV_loss = cur_model.val_y(tf.cast(input_data, tf.float32),  tf.cast(y, tf.float32), val_idx)

        sum_loss += loss
        sum_CV_loss += CV_loss

        if (i+1) % 20 == 0:
            if len(losslist_CV) > 0 and sum_CV_loss / 20 >= losslist_CV[-1]:
                count += 1
            else:
                count = 0
        
            if flag_early_stop:
                if i > 400 and count >= 1:
                    break
            
            losslist.append(sum_loss / 20)
            losslist_CV.append(sum_CV_loss / 20)

            sum_loss = 0
            sum_CV_loss = 0

    return cur_model

 
def save_my_model(save_path, save_name, need_save_model):
    """
    Save the model weights to a specified path.

    Args:
        save_path (str): Directory to save the model weights.
        save_name (str): Filename for the saved weights.
        need_save_model (tf.keras.Model): Model instance to be saved.

    Returns:
        None
    """
    path = save_path + '/' + save_name

    need_save_model.save_weights(path)
    print("Already saved the model's weights in file" + path)


def load_my_model(load_path, load_name, need_load_model, config, activation, adjs):
    """
    Load a saved model from a specified path.

    Args:
        load_path (str): Directory where the model is saved.
        load_name (str): Filename of the saved model.
        need_load_model (tf.keras.Model): Model class to instantiate.
        config (dict): Model configuration parameters.
        activation (tf activation function): Activation function.

    Returns:
        tf.keras.Model: Loaded model instance.
    """
    model = need_load_model(config, activation, adjs)

    path = load_path + '/' + load_name

    model.load_weights(path)
    print("Model successfully loaded.")

    return model


def find_hyperparameter(set_configs, data_name, Model_name, activation):
    data = load_data(data_name)
    train_idx, val_idx, test_idx = split_train_val_test(data[0][0], 0.7, 0.15, 0.15)

    for i in range(0, 10):
        cur_all_input, yf, mu1, mu0, adjs = data_preparation(data_name, i, data)

        true_ite = mu1 - mu0
        val_true_ite = true_ite[val_idx]
        val_true_ate = np.mean(true_ite[val_idx])
        test_true_ite = true_ite[test_idx]
        test_true_ate = np.mean(true_ite[test_idx])

        for j in range(len(set_configs)):
            cur_val_results = []
            config = set_configs[j]
            cur_model = train(
                Model_name, 
                tf.cast(cur_all_input, tf.float32), 
                tf.cast(yf, tf.float32), 
                train_idx, 
                val_idx, 
                config, 
                config["iterations"], 
                config["lr_rate"], 
                config["lr_weigh_decay"], 
                config["flag_early_stop"], 
                activation=activation, 
                true_ite=val_true_ite, 
                cur_adj=adjs
            )

            cur_save_model_name = "model"
            cur_save_path = './save_Models_HINITE/Model_' + data_name + '_' + str(Model_name)[8:-2] + "_split_" + str(i)
            if config['rep_alpha'] > 0:
                cur_save_path += '_rep_alpha' + str(config['rep_alpha'])

            os.makedirs(cur_save_path, exist_ok=True)
            save_my_model(cur_save_path, cur_save_model_name, cur_model)


def save_results(save_result, save_name):
	np.save(save_name, save_result) 
	print("saved all results ")


def load_data(data_name):
    data = []

    if data_name == 'Youtube':
        all_x = []
        all_t = []
        all_yf = []
        all_m1 = []
        all_m0 = []
        all_adjs = []
        for i in range(10):
            cur_name = "./data/Youtube/Youtube_"
            cur_name_x = cur_name + "x_" + str(i)+ ".npy"
            cur_name_t = cur_name + "T_" + str(i)+ ".npy"
            cur_name_yf = cur_name + "yf_" + str(i) + ".npy"
            cur_name_m1 = cur_name + "y1_spe_" + str(i) + ".npy"
            cur_name_m0 = cur_name + "y0_spe_" + str(i) + ".npy"
            cur_name_adj = cur_name + "adjs_" + str(i) + ".npy"

            cur_x = np.load(cur_name_x)
            cur_t = np.load(cur_name_t)
            cur_yf = np.load(cur_name_yf)
            cur_m1 = np.load(cur_name_m1)
            cur_m0 = np.load(cur_name_m0)
            cur_adj = np.load(cur_name_adj)

            all_x.append(cur_x)
            all_t.append(cur_t)
            all_yf.append(cur_yf)
            all_m1.append(cur_m1)
            all_m0.append(cur_m0)
            all_adjs.append(cur_adj)
            
    elif data_name == 'Flk':
        all_x = []
        all_t = []
        all_yf = []
        all_m1 = []
        all_m0 = []
        all_adjs = []
        for i in range(10):
            cur_name = "./data/flk/Flickr"
            cur_name_x = cur_name + "_" + str(i) + "_x.npy"
            cur_name_t = cur_name + "_" + str(i)+ "_T.npy"
            cur_name_yf = cur_name + "_" + str(i) + "_yf.npy"
            cur_name_m1 = cur_name + "_" + str(i) + "_y1_spe.npy"
            cur_name_m0 = cur_name + "_" + str(i) + "_y0_spe.npy"
            cur_name_adj = cur_name + "_" + str(i) + "_adjs.npy"

            cur_x = np.load(cur_name_x)
            cur_t = np.load(cur_name_t)
            cur_yf = np.load(cur_name_yf)
            cur_m1 = np.load(cur_name_m1)
            cur_m0 = np.load(cur_name_m0)
            cur_adj = np.load(cur_name_adj)

            all_x.append(cur_x)
            all_t.append(cur_t)
            all_yf.append(cur_yf)
            all_m1.append(cur_m1)
            all_m0.append(cur_m0)
            all_adjs.append(cur_adj)
            
    data = [all_x, all_t, all_yf, all_m1, all_m0, all_adjs]

    return data


def data_preparation(data_name, split_idx, data):

    all_x, all_t, all_yf, all_m1, all_m0, all_adj = data

    cur_x = all_x[split_idx]
    cur_t = all_t[split_idx].reshape(len(cur_x), 1)
    cur_yf = all_yf[split_idx].reshape(len(cur_x), 1)
    cur_m1 = all_m1[split_idx].reshape(len(cur_x), 1)
    cur_m0 = all_m0[split_idx].reshape(len(cur_x), 1)
    cur_adj = all_adj[split_idx]

    cur_all_input = np.concatenate([cur_x, cur_t], 1)

    return cur_all_input, cur_yf, cur_m1, cur_m0, cur_adj


def config_pare_HINITE(
        iterations,
        lr_rate,
        lr_weigh_decay,
        flag_early_stop,
        use_batch,
        rep_alpha,
        flag_norm_gnn,
        flag_norm_rep,
        out_dropout,
        GNN_dropout,
        rep_dropout,
        inp_dropout,
        rep_hidden_layer,
        rep_hidden_shape,
        GNN_hidden_layer,
        GNN_hidden_shape,
        head_num_att,
        out_T_layer,
        out_C_layer,
        out_hidden_shape,
        GNN_alpha,
        divide,
        hete_att_size
    ):

        all_configs = []

        for i in range(len(rep_alpha)):
            config = {
                "iterations": iterations,
                "lr_rate": lr_rate,
                "lr_weigh_decay": lr_weigh_decay,
                "flag_early_stop": flag_early_stop,
                "rep_alpha": rep_alpha[i],
                "flag_norm_gnn": flag_norm_gnn,
                "flag_norm_rep": flag_norm_rep,
                "out_dropout": out_dropout,
                "GNN_dropout": GNN_dropout,
                "rep_dropout": rep_dropout,
                "inp_dropout": inp_dropout,
                "use_batch": use_batch,
                "rep_hidden_layer": rep_hidden_layer,
                "rep_hidden_shape": rep_hidden_shape,
                "GNN_hidden_layer": GNN_hidden_layer,
                "GNN_hidden_shape": GNN_hidden_shape,
                "head_num_att": head_num_att,
                "out_T_layer": out_T_layer,
                "out_C_layer": out_C_layer,
                "out_hidden_shape": out_hidden_shape,
                "GNN_alpha": GNN_alpha,
                "divide": divide,
                "hete_att_size": hete_att_size
            }

            all_configs.append(config)

        return all_configs
