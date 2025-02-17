import tensorflow as tf
from tensorflow import keras

import HINITE
import ourlayers
import utils


def main_YB_HINITE():
    configs = utils.config_pare_HINITE(
        iterations=2000,
        lr_rate=0.001,
        lr_weigh_decay=0.001,
        flag_early_stop=True,
        use_batch=512,
        rep_alpha=[0.01, 0.1, 0.5, 1.0, 1.5],
        flag_norm_gnn=False,
        flag_norm_rep=False,
        out_dropout=0.2,
        GNN_dropout=0.2,
        rep_dropout=0.2,
        inp_dropout=0.0,
        rep_hidden_layer=3,
        rep_hidden_shape=[128, 64, 64],
        GNN_hidden_layer=3,
        GNN_hidden_shape=[64, 64, 32], 
        divide=True,
        head_num_att=1,
        out_T_layer=3,
        out_C_layer=3,
        out_hidden_shape=[128, 64, 32],
        GNN_alpha=0.0,
        hete_att_size=[128, 128, 64]
    )
    activation = tf.nn.relu

    utils.find_hyperparameter(
        set_configs=configs, 
        data_name='Youtube',
        Model_name=HINITE.HINITE,
        activation=activation
    )


if __name__ == '__main__':
    main_YB_HINITE()