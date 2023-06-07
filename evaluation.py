import numpy as np   
import tensorflow as tf 

def policy_val(t, yf, eff_pred):
    """ Computes the value of the policy defined by predicted effect """
    
    
    eff_pred = np.array(eff_pred)
    t = np.array(t)
    yf = np.array(yf)
    if np.any(np.isnan(eff_pred)):
        return np.nan, np.nan

    policy = eff_pred>0
    treat_overlap = (policy==t)*(t>0)
    control_overlap = (policy==t)*(t<1)

    if np.sum(treat_overlap)==0:
        treat_value = 0
    else:
        treat_value = np.mean(yf[treat_overlap])

    if np.sum(control_overlap)==0:
        control_value = 0
    else:
        control_value = np.mean(yf[control_overlap])

    pit = np.mean(policy)
    policy_value = pit*treat_value + (1-pit)*control_value

    return policy_value


def evaluate_att_policy(Model,inputtensor,test_idx,true_att,true_y,RCT_flags):

    #print("RCT_flags",RCT_flags)
    pre_T,pre_C = Model.pre_no_interf(inputtensor,test_idx,only_T=True)
    
    all_t = inputtensor[:,-1]

    t_test = tf.gather(all_t,test_idx)
    y_test = tf.gather(true_y,test_idx)


    only_t_ITE = pre_T - pre_C
    ATT = tf.reduce_mean(only_t_ITE)

    #print("ATT",ATT)

    pre_1,pre_0 = Model.pre_no_interf(inputtensor,test_idx,only_T=False)
    ITE = pre_1 - pre_0
    #print("ITE",ITE.shape)
    #print("RCT_flags",RCT_flags)
    test_rct_flags = np.array(RCT_flags)[test_idx]
    #print("test_rct_flags",test_rct_flags)
    rct_idx = tf.where(test_rct_flags>0)[:,0]
    #print("rct_idx ",rct_idx)
    t_rct = np.array(tf.gather(t_test,rct_idx))
    t_rct = t_rct.reshape(len(t_rct),)
    y_rct = np.array(tf.gather(y_test,rct_idx))
    y_rct = y_rct.reshape(len(y_rct),)
    rct_ITE = np.array(tf.gather(ITE,rct_idx))
    rct_ITE = rct_ITE.reshape(len(rct_ITE),)
    print("t shape",t_rct.shape)
    print("y shape",y_rct.shape)
    print("ite shape",rct_ITE.shape)

    pol = policy_val(t_rct , y_rct, rct_ITE)
    pol_risk = 1-pol
    print("pol",pol)


    err_att =np.abs(ATT-true_att)
    print("pol_risk",pol_risk)
    print("error att",err_att)
                              
    return pol_risk,err_att

def evaluate_ate_pehe(Model,inputtensor,test_idx,true_ite,true_ate):

    pre_T,pre_C = Model(tf.cast(inputtensor,tf.float32),test_idx)
    
    ITE = pre_T - pre_C


    ATE = tf.reduce_mean(ITE)

    pehe = np.mean((ITE - true_ite)**2)
    err_ate = np.abs(ATE-true_ate)
                              
    return pehe,err_ate


def evalate(Model,inputtensor,test_idx,true_ate,true_y, true_ite = [],RCT_flags=None):
    if len(true_ite) > 0:
        pehe, err_ate = evaluate_ate_pehe(Model,inputtensor,test_idx,true_ite,true_ate)
        return pehe,err_ate
    else:
        pol_risk,err_att = evaluate_att_policy(Model,inputtensor,test_idx,true_ate,true_y,RCT_flags)
        return pol_risk,err_att

