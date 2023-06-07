import numpy as np   
import tensorflow as tf 

def evaluate_ate_pehe(Model,inputtensor,test_idx,true_ite,true_ate):

    pre_T,pre_C = Model(tf.cast(inputtensor,tf.float32),test_idx)
    
    ITE = pre_T - pre_C


    ATE = tf.reduce_mean(ITE)

    pehe = np.mean((ITE - true_ite)**2)
    err_ate = np.abs(ATE-true_ate)
                              
    return pehe,err_ate


def evalate(Model,inputtensor,test_idx,true_ate,true_y, true_ite = [],RCT_flags=None):
    
    pehe, err_ate = evaluate_ate_pehe(Model,inputtensor,test_idx,true_ite,true_ate)
    return pehe,err_ate
   
