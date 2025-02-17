import math
import random

import tensorflow as tf
from tensorflow import keras

import utils
import ourlayers


class HINITE(keras.Model):
    def __init__(self, config, activation=tf.nn.relu, init_adj=None):
        super(HINITE, self).__init__()
        print("Initialization ...")

        self.rep_layers = []
        self.gnn_layers = []
        self.out_T_layers = []
        self.out_C_layers = []
        self.out_layers = []
        self.init_adjs = []
        self.train_loss = None

        for i in range(len(init_adj)):
        	temp_adj_plus_self = init_adj[i] + tf.eye(init_adj[i].shape[0])
        	self.init_adjs.append(temp_adj_plus_self)

        self.activation = activation
        self.rep_alpha = config['rep_alpha']
        self.flag_norm_gnn= config['flag_norm_gnn']
        self.flag_norm_rep= config['flag_norm_rep']
        self.out_dropout = config['out_dropout']
        self.GNN_dropout = config['GNN_dropout']
        self.rep_dropout = config['rep_dropout']
        self.inp_drop = config['inp_dropout']
        self.use_batch = config['use_batch']
        self.optimizer = keras.optimizers.Adam(lr=config['lr_rate'], decay=config['lr_weigh_decay'])
        
        for i in range(config['rep_hidden_layer']):
            h = ourlayers.RepLayer(config['rep_hidden_shape'][i], activation=self.activation)
            self.rep_layers.append(h)
        
     
        for i in range(config['GNN_hidden_layer']):
            g = ourlayers.HXGATlayer(
                att_embedding_size=config['GNN_hidden_shape'][i], 
                hete_att_size=config['hete_att_size'][i], 
                hete_num=len(init_adj), 
                reduction="mean", 
                activation=self.activation, 
                use_bias=False, 
                divide=False
            )
            self.gnn_layers.append(g)

        for i in range(config['out_T_layer']):
            out_T = keras.layers.Dense(config['out_hidden_shape'][i], activation=self.activation, kernel_initializer=tf.keras.initializers.glorot_uniform())
            self.out_T_layers.append(out_T)
   
        for i in range(config['out_C_layer']):
            out_C = keras.layers.Dense(config['out_hidden_shape'][i], activation=self.activation, kernel_initializer=tf.keras.initializers.glorot_uniform())
            self.out_C_layers.append(out_C)
       
        self.layer_6_T = keras.layers.Dense(1)
        self.layer_6_C = keras.layers.Dense(1)

    def call(self, input_tensor, idxs, training = False):
        input_x = input_tensor[:, :-1]
        input_t = tf.constant(input_tensor[:, -1],shape = [input_x.shape[0], 1])
        
        hidden = input_x
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden, flag=training)
        if self.flag_norm_rep:
            h_rep_norm = hidden / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(hidden), axis=1, keepdims=True), 1e-10, np.inf))
        else:
            h_rep_norm = hidden * 1.0
            
        concat_rep_t = tf.concat([h_rep_norm, input_t], axis=1)

        GNN = concat_rep_t
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i]([GNN, self.init_adjs])          
        if self.flag_norm_gnn:
            GNN_norm = GNN / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(GNN), axis=1, keepdims=True), 1e-10, np.inf))
        else: 
            GNN_norm = GNN * 1.0

        concated_data = tf.concat([h_rep_norm, GNN_norm ], axis=1)

        gathered_concated_data = tf.gather(concated_data, idxs)
       
        outnn_T = gathered_concated_data
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.layer_6_T(outnn_T)

        outnn_C = gathered_concated_data
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.layer_6_C(outnn_C)
    
        return output_T, output_C
    
    def get_loss(self, input_tensor, all_y, train_idx, training=True):

        input_x = input_tensor[:, :-1]
        input_t = tf.constant(input_tensor[:, -1], shape=[input_x.shape[0], 1])
        
        input_x = tf.nn.dropout(input_x, self.inp_drop)
        
        hidden = input_x
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden, flag=training)
            hidden = tf.nn.dropout(hidden, self.rep_dropout)
        if self.flag_norm_rep:
            h_rep_norm = hidden / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(hidden), axis=1, keepdims=True), 1e-10, np.inf))
        else:
            h_rep_norm = hidden * 1.0
            
        concat_rep_t = tf.concat([h_rep_norm, input_t], axis=1)

        GNN = concat_rep_t
       
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i]([GNN, self.init_adjs])    
            GNN = tf.nn.dropout(GNN, self.GNN_dropout)
        if self.flag_norm_gnn:
            GNN_norm = GNN / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(GNN), axis=1, keepdims=True), 1e-10, np.inf))
        else: 
            GNN_norm = GNN*1.0

        concated_data = tf.concat([h_rep_norm, GNN_norm], axis=1)

        train_concated_data = tf.gather(concated_data, train_idx)
        train_input_t = tf.gather(input_t, train_idx)
        train_y = tf.gather(all_y, train_idx)
        train_hidden = tf.gather(h_rep_norm, train_idx)
        train_GNN = tf.gather(GNN_norm, train_idx)

        if self.use_batch:
            batch = random.sample(range(0, len(train_concated_data)), self.use_batch)
            train_concated_data = tf.gather(train_concated_data, batch)
            train_input_t = tf.gather(train_input_t, batch)
            train_y = tf.gather(train_y, batch)
            train_hidden = tf.gather(train_hidden, batch)
            train_GNN = tf.gather(train_GNN, batch)

        group_t, group_c, i_0, i_1= utils.divide_t_c(train_concated_data, train_input_t)

        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
            outnn_T = tf.nn.dropout(outnn_T, self.out_dropout)
        output_T = self.layer_6_T(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
            outnn_C =  tf.nn.dropout(outnn_C, self.out_dropout)
        output_C = self.layer_6_C(outnn_C)

        y_pre = tf.dynamic_stitch([i_0, i_1], [output_C, output_T])
        
        pred_error = tf.reduce_mean(tf.square(train_y - y_pre))   
        print("Train loss", pred_error)

        rep_error = self.rep_alpha * utils.COMP_hsic(train_hidden, train_input_t)
        print("hsic rep_loss", rep_error)

        L =   rep_error + pred_error 
        print("total loss", L)

        return L
    
   
    
    def get_grad(self, input_tensor, y, train_idx):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            L = self.get_loss(input_tensor, y, train_idx)
            self.train_loss = L
            g = tape.gradient(L, self.variables)

        return g

    def network_learn(self, input_tensor, y, train_idx):
        g = self.get_grad(input_tensor, y, train_idx)
        self.optimizer.apply_gradients(zip(g, self.variables))

        return self.train_loss

    def val_y(self, input_tensor, all_y, idxs, training=False):
        input_x = input_tensor[:, :-1]
        input_t = tf.constant(input_tensor[:,-1], shape=[input_x.shape[0], 1])

        hidden = input_x
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden, flag=training)
        if self.flag_norm_rep:
            h_rep_norm = hidden / tf.sqrt(tf.clip_by_value(tf.reduce_sum(tf.square(hidden), axis=1, keepdims=True), 1e-10, np.inf))
        else:
            h_rep_norm = hidden * 1.0

        concat_rep_t = tf.concat([h_rep_norm, input_t], axis=1)

        GNN = concat_rep_t
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i]([GNN, self.init_adjs])
        if self.flag_norm_gnn:
            GNN_norm = GNN / tf.sqrt( tf.clip_by_value(tf.reduce_sum(tf.square(GNN), axis=1, keepdims=True), 1e-10, np.inf))
        else:
            GNN_norm = GNN * 1.0

        concated_data = tf.concat([h_rep_norm, GNN_norm], axis=1)    

        gathered_concated_data = tf.gather(concated_data,idxs)
        gathered_input_t = tf.gather(input_t, idxs)
        gathered_y = tf.gather(all_y, idxs)

        group_t,group_c,i_0,i_1 = utils.divide_t_c(gathered_concated_data, gathered_input_t)
        p = tf.divide(tf.reduce_sum(gathered_input_t), gathered_input_t.shape[0])
        y_T = tf.gather(gathered_y, i_1)
        y_C = tf.gather(gathered_y, i_0)

        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.layer_6_T(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.layer_6_C(outnn_C)

        y_pre = tf.dynamic_stitch([i_0, i_1], [output_C, output_T])
        
        pred_error = tf.reduce_mean(tf.square(train_y - y_pre)) 

        return pred_error