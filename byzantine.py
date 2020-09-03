import mxnet as mx
import numpy as np
from copy import deepcopy
import time
from numpy import random
from mxnet import nd, autograd, gluon
import math
import nd_aggregation
import pdb

device = mx.cpu()

def no_byz(epoch, v, net, f, lr):
    return v

def partial_trim(epoch, v, net, f):
    # apply partial knowledge trimmed mean attack
    
    vi_shape = v[0].shape
    
    #first compute the distribution parameters
    all_grads = nd.concat(*v, dim=1)
    adv_grads = all_grads[:, :f]
    e_mu = nd.mean(adv_grads, axis=1) # mean
    e_sigma = nd.sqrt(nd.sum(nd.square(nd.subtract(adv_grads, e_mu.reshape(-1,1))), axis=1) / f) # standard deviation

    for i in range(f):
        # apply attack to compromised worker devices with randomness
        v[i] = (e_mu - nd.multiply(e_sigma, nd.sign(e_mu)) * (3. + nd.random.uniform(shape=e_sigma.shape))).reshape(vi_shape)               
            
    return v   

    
def full_trim(epoch, v, net, f):
    # apply full knowledge trimmed mean attack
    vi_shape = v[0].shape
    v_tran = nd.concat(*v, dim=1)
    maximum_dim = nd.max(v_tran, axis=1).reshape(vi_shape)
    minimum_dim = nd.min(v_tran, axis=1).reshape(vi_shape)
    direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
    #direction = old_direction
    directed_dim = (direction > 0) * minimum_dim + (direction < 0) * maximum_dim

    for i in range(f):
        random_12 = 1 + nd.random.uniform(shape=vi_shape)
        v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    return v     

def full_krum(epoch, v, net, f, lr):

    prob = 1
    adapt = mx.nd.array(np.random.choice(2, len(v[0]), p=[1-prob, prob])).as_in_context(device)
    if (f==0):
        return v
    e = 0.00001/len(v[0])
    direction = nd.sign(nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True))
    l_max = lambda_max(epoch, v, net, f, lr)
    l = find_lambda(l_max, v, direction, len(v), f, lr)
    #print (l)
    if (l>0):
        v[0] = -(direction*l)/lr
        for i in range(1, f):
            v[i] = mx.nd.random.uniform(v[0]-e, v[0]+e)
    return v

def lambda_max(epoch, v, net, f, lr ): #(m, c, params, global_param):

    m = len(v)
    dist = mx.nd.zeros((m,m)).as_in_context(device)
    for i in range (0, m):
        for j in range(0, i):
            dist[i][j] = nd.norm(v[i] - v[j])*lr
            dist[j][i] = dist[i][j]   
    sorted_benign_dist = mx.nd.sort(dist[f:,f:])
    sum_benign_dist = mx.nd.sum(sorted_benign_dist[:, :(m-f-1)], axis=1)
    min_distance = mx.nd.min(sum_benign_dist).asscalar()
    
    dist_global = mx.nd.zeros(m-f).as_in_context(device)
    for i in range(f, m):
        dist_global[i-f] = nd.norm(v[i])*lr
    max_global_dist = mx.nd.max(dist_global).asscalar()
    scale = 1.0/(len(v[0]))
    return (math.sqrt(scale)/(m-2*f-1))*min_distance + math.sqrt(scale)*max_global_dist


def find_lambda(lambda_current, params, s, m, c, lr):
    
    print (lambda_current)
    if (lambda_current <= 0.00001):
        return 0.0
 
    #params_local = params[:m].copy()
    params_local = params.copy()
    params_local[0][:] = -(lambda_current)*s/lr #[:] is important
    for i in range(1, c):
        params_local[i] = params_local[0]
    model_selected = local_krum(params_local, c)
    #print (model_selected, lambda_current)
    if (model_selected <= c):
        return lambda_current
    else:
        return find_lambda(lambda_current*0.5, params, s, m, c, lr)
    
def local_krum(param_list, f):
    print("Local krum")
    #param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    k = len(param_list) - f - 2
    dist = mx.nd.zeros((len(param_list),len(param_list)))
    for i in range (0, len(param_list)):
        for j in range(0, i):
            dist[i][j] = nd.norm(param_list[i] - param_list[j])
            dist[j][i] = dist[i][j]      
    sorted_dist = mx.nd.sort(dist)
    sum_dist = mx.nd.sum(sorted_dist[:,:k+1], axis=1)
    model_selected = mx.nd.argmin(sum_dist).asscalar().astype(int)        
    
    return model_selected    
        
'''
def krum_attack(active, params, global_param, s, m, c):
    
    prob = 1
    adapt = mx.nd.array(np.random.choice(2,P, p=[1-prob, prob])).as_in_context(device)
    if (c==0):
        return -1, 0
    e = 0.00001/len(params[0])
    l_max = lambda_max(m, c, params, global_param)
    l = find_lambda(l_max, params, global_param, s, m, c, adapt, active)
    #l = 0.004
    if (l == 0):
        return 0, 0

    attacked = np.random.choice(c, 1)[0]
    params[attacked] = params[attacked] - l*s*adapt
    for i in range(0,c):
        if (i != attacked):
            params[i] = mx.nd.random.uniform(params[attacked]-e, params[attacked]+e)
    return attacked, l
'''
        
