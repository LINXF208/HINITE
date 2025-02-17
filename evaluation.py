import numpy as np   
import tensorflow as tf 


def evalate(model, input_tensor, test_idx, true_ate, true_y, true_ite):
    pre_y1, pre_y0 = model(tf.cast(input_tensor, tf.float32), test_idx)

    pre_ite = pre_y1 - pre_y0
    pre_ate = tf.reduce_mean(pre_ite)

    pehe = np.mean((pre_ite - true_ite)**2)
    err_ate = np.abs(pre_ate -true_ate)
    
    pehe, err_ate = evaluate_ate_pehe(model, input_tensor, test_idx, true_ite, true_ate)
    
    return pehe, err_ate
