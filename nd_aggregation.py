import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
from copy import deepcopy
import time
import pdb

# trimmed mean
def trim(epoch, gradients, net, lr, byz, old_direction, f = 0, b = 0):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, f, b)
    direction = nd.sign(nd.sum(nd.concat(*param_list, dim=1), axis=-1, keepdims=True)) 
    flip_count = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    n = len(param_list)
    m = n - b*2
    trim_nd = nd.mean(sorted_array[:, b:(b+m)], axis=-1, keepdims=1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    return trim_nd, direction, flip_count

def krum(epoch, gradients, net, lr, byz, old_direction, f = 0, b = 0):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, f, lr)
    direction = nd.sign(nd.sum(nd.concat(*param_list, dim=1), axis=-1, keepdims=True))
    flip_count = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
    k = len(param_list) - f - 2
    dist = mx.nd.zeros((len(param_list),len(param_list)))
    for i in range (0, len(param_list)):
        for j in range(0, i):
            dist[i][j] = nd.norm(param_list[i] - param_list[j])
            dist[j][i] = dist[i][j]    
            
    sorted_dist = mx.nd.sort(dist)
    sum_dist = mx.nd.sum(sorted_dist[:,:k+1], axis=1)
    model_selected = mx.nd.argmin(sum_dist).asscalar().astype(int)   
    #if (model_selected < f):
    #    lr = 1.0
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * param_list[model_selected][idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size  
    return model_selected, direction, flip_count