import mxnet as mx
import numpy as np
from copy import deepcopy
import time
from numpy import random
from mxnet import nd, autograd, gluon
import math
import pdb

def no_byz(epoch, v, net, f, lr, active, max_flip = 1.0):
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

    
def full_trim(epoch, v, net, f, lr, active, max_flip = 1.0):
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
        if (active[0] < f):
            v[i] = directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12)
    #pdb.set_trace()
    return v     

def full_krum(epoch, v, net, f, lr, active, max_flip = 1.0):

    if (f==0):
        return v
    e = 0.00001/len(v[0])
    avg_grads = nd.sum(nd.concat(*v, dim=1), axis=-1, keepdims=True)
    direction = nd.sign(avg_grads)
    topk = nd.argsort(nd.abs(avg_grads).reshape(-1))
    n_flips = int(max_flip*len(v[0]))
    current_f = len(np.where(np.where(active<f)[0]<f)[0])
    l_max = lambda_max(epoch, v, net, current_f, lr)
    l = find_lambda(l_max, v, direction, len(v), current_f, lr, topk, max_flip)
    print (l)
    if (l>0 and active[0] < f):
        v[0][topk[-n_flips:]] = -(direction[topk[-n_flips:]]*l)/lr
        for i in range(1, f):
            if (active[i] < f):
                v[i] = mx.nd.random.uniform(v[0]-e, v[0]+e)
    return v

    
def lambda_max(epoch, v, net, f, lr ): #(m, c, params, global_param):

    if (f == 0):
        return 0.0
    m = len(v)
    dist = mx.nd.zeros((m,m))
    for i in range (0, m):
        for j in range(0, i):
            dist[i][j] = nd.norm(v[i] - v[j])*lr
            dist[j][i] = dist[i][j]   
    sorted_benign_dist = mx.nd.sort(dist[f:,f:])
    sum_benign_dist = mx.nd.sum(sorted_benign_dist[:, :(m-f-1)], axis=1)
    min_distance = mx.nd.min(sum_benign_dist).asscalar()
    
    dist_global = mx.nd.zeros(m-f)
    for i in range(f, m):
        dist_global[i-f] = nd.norm(v[i])*lr
    max_global_dist = mx.nd.max(dist_global).asscalar()
    scale = 1.0/(len(v[0]))
    return (math.sqrt(scale)/(m-2*f-1))*min_distance + math.sqrt(scale)*max_global_dist


def find_lambda(lambda_current, params, s, m, c, lr, topk, max_flip=1.0):
    
    if (lambda_current <= 0.00001):
        return 0.0
    n_flips = int(max_flip*len(params[0]))
    params_local = params.copy()
    if (max_flip==1.0):
        params_local[0][topk[-n_flips:]] = -(lambda_current)*s[topk[-n_flips:]]/lr #[:] is important
    for i in range(1, c):
        params_local[i] = params_local[0]
    model_selected = local_krum(params_local, c)
    if (model_selected <= c):
        del params_local
        return lambda_current
    else:
        del params_local
        return find_lambda(lambda_current*0.5, params, s, m, c, lr, topk, max_flip)
    
def local_krum(param_list, f):

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
        
