import torch
from copy import deepcopy
import math
import numpy as np
import copy
import pdb

##No attack
def benign(device, lr, param_list, cmax):

    return param_list

###121421-2050: Adding the agnostic attack

def shej_agnostic(device, lr, param_list, cmax, dev_type='unit_vec'):

    all_updates = param_list[cmax:]       
    model_re = torch.mean(all_updates, 0)  
    n_attackers = cmax
    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).float().cuda()

    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0
    
    distances = []
    for update in all_updates:
        distance = torch.norm((all_updates - update), dim=1) ** 2
        distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
    
    max_distance = torch.max(distances)
    del distances
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        distance = torch.norm((all_updates - mal_update), dim=1) ** 2
        max_d = torch.max(distance)
        
        if max_d <= max_distance:

            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)                                                                                                                   
    mal_updates = torch.cat((mal_updates, all_updates), 0) 
    return mal_updates


def tr_mean(all_updates, n_attackers):
    sorted_updates = torch.sort(all_updates, 0)[0]
    out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates)
    return out

def shej_attack(device, lr, param_list, cmax, dev_type='unit_vec'):
     
    all_updates = param_list[cmax:]
    model_re = torch.mean(all_updates, 0)
    n_attackers = cmax

    if dev_type == 'unit_vec':
        deviation = model_re / torch.norm(model_re)  # unit vector, dir opp to good dir
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(all_updates, 0)

    lamda = torch.Tensor([10.0]).cuda() 
    threshold_diff = 1e-5
    prev_loss = -1
    lamda_fail = lamda
    lamda_succ = 0
    iters = 0     
    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = (model_re - lamda * deviation)
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, all_updates), 0)

        agg_grads = tr_mean(mal_updates, n_attackers)

        loss = torch.norm(agg_grads - model_re)

        if prev_loss < loss:
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2
        prev_loss = loss

    mal_update = (model_re - lamda_succ * deviation)
    mal_updates = torch.stack([mal_update] * n_attackers)
    mal_updates = torch.cat((mal_updates, all_updates), 0)

    return mal_updates    

##Full-trim attack
def full_trim(device, lr, param_list, cmax):#, old_direction):

    max_dim = torch.max(-param_list, axis=0)[0]
    min_dim = torch.min(-param_list, axis=0)[0]
    direction = torch.sign(torch.sum(-param_list, axis=0)).to(device) #estimated benign direction
    directed_dim = (direction > 0) * min_dim + (direction < 0) * max_dim
    for i in range(cmax):
        random_12 = 1 + torch.rand(len(param_list[0])).to(device)
        param_list[i] = -(directed_dim * ((direction * directed_dim > 0) / random_12 + (direction * directed_dim < 0) * random_12))
    return param_list

##Full-krum attack
def full_krum(device, lr, v, f):#, old_direction):

    if (f==0):
        return v
    e = 0.0001/len(v[0]) ##Noise value to be added by the malicious clients
    direction = torch.sign(torch.sum(v, axis=0)) #estimated direction
    l_max = lambda_max(device, v, f) #maximum theoretical attack magnitude
    l = find_lambda(device, l_max, v, direction, len(v), f) #optimized attack magnitude computed
    #print ("Lambda selected: ", l)
    if (l>0):
        v[0] = -(direction*l) #go in the opposite direction of the benign gradients
        #malicious clients copy the same gradients with noise
        for i in range(1, f):
            noise = torch.FloatTensor(1, len(v[0])).uniform_(-e, e).to(device)
            v[i] = v[0] + noise
    return v

#compurtes the maximum theoretical attack magnitude for full-krum
def lambda_max(device, v, f): #(m, c, params, global_param):

    m = len(v)
    dist = torch.zeros((m,m)).to(device)
    for i in range (0, m):
        for j in range(0, i):
            dist[i][j] = torch.norm(v[i] - v[j])
            dist[j][i] = dist[i][j]   
    sorted_benign_dist = torch.sort(dist[f:,f:])
    sum_benign_dist = torch.sum((sorted_benign_dist[0][:, :(m-f-1)])**2, axis=1)
    min_distance = torch.min(sum_benign_dist).item()
    
    dist_global = torch.zeros(m-f).to(device)
    for i in range(f, m):
        dist_global[i-f] = torch.norm(v[i])
    max_global_dist = torch.max(dist_global).item()
    scale = 1.0/(len(v[0]))
    return (math.sqrt(scale/(m-2*f-1))*min_distance) + math.sqrt(scale)*max_global_dist

##finds the optimal attack magnitude for full-krum
def find_lambda(device, lambda_current, params, s, m, c):
    
    if (lambda_current <= 0.00001):
        return 0.0
 
    params_local = params.detach().clone()
    params_local[0][:] = -(lambda_current)*s #[:] is important
    for i in range(1, c):
        params_local[i] = deepcopy(params_local[0])
    model_selected = local_krum(device, params_local, c)
    if (model_selected <= c):
        del params_local
        return lambda_current
    else:
        del params_local
        return find_lambda(device, lambda_current*0.5, params, s, m, c)
   
##Malicious clients run a local krum aggregation to make sure they succeed
def local_krum(device, param_list, f):

    k = len(param_list) - f - 2
    dist = torch.zeros((len(param_list),len(param_list))).to(device)
    for i in range (0, len(param_list)):
        for j in range(0, i):
            dist[i][j] = torch.norm(param_list[i] - param_list[j])
            dist[j][i] = dist[i][j]      
    sorted_dist = torch.sort(dist)
    sum_dist = torch.sum(sorted_dist[0][:,:k+1], axis=1)
    model_selected = torch.argmin(sum_dist).item()        
    
    return model_selected    

##Adaptive trim attack
def adaptive_trim(device, lr, param_list, old_direction, nbyz, fs_min, fs_max, weight):

    max_dim = torch.max(-param_list, axis=0)[0]
    min_dim = torch.min(-param_list, axis=0)[0]

    ben_grads = torch.mean(-param_list, axis=0).to(device)
    ben_direction = torch.sign(ben_grads).to(device)
    directed_dim = (ben_direction > 0) * min_dim + (ben_direction < 0) * max_dim

    random_l2 = 1 + torch.rand(len(param_list[0])).to(device)
    ##Original attack
    target_attack = -(directed_dim * ((ben_direction * directed_dim > 0) / random_l2 + (ben_direction * directed_dim < 0) * random_l2))

    direction = torch.sign(target_attack)
    flip = torch.sign(direction*(direction-old_direction.reshape(-1)))
    flip_score = torch.sum(flip*(target_attack**2))
    ##Finding gradients with largest updates to attack
    topk = torch.argsort(torch.abs(ben_grads).reshape(-1))

    for i in range(nbyz):
        random_l2 = 1 + torch.rand(len(param_list[0])).to(device)
        param_list[i] = (-(directed_dim * ((direction * directed_dim > 0) / random_l2 + (direction * directed_dim < 0) *random_l2)))
    if (flip_score < fs_max):
        #already stealthy enough
        return param_list
    else: ##Need to dilute the attack to become stealthy
        for i in range(nbyz):
            test_attack = deepcopy(param_list[i])
            step = 5 #undo attack on 5% of the gradients at a time
            for perc in range (0, 100, step):
                start_idx = round((perc/100)*len(topk))
                end_idx = round(((perc+step)/100)*len(topk))
                test_attack[topk[start_idx:end_idx]] = ben_grads[topk[start_idx:end_idx]]
                direction = torch.sign(test_attack)
                flip = torch.sign(direction*(direction-old_direction.reshape(-1)))
                flip_score = torch.sum(flip*(test_attack**2))
                if (flip_score < fs_max):
                    diff = weight[i]*(param_list[i] - test_attack)  ##targeted attack carried on to other clients
                    param_list[i] = deepcopy(test_attack)
                    
                    if (i+1 < nbyz):
                        param_list[i+1] = param_list[i+1] + diff/weight[i+1]
                        fs_rem = torch.sum((torch.sign(torch.sign(param_list[i+1])*(torch.sign(param_list[i+1])-old_direction.reshape(-1))))*(param_list[i+1]**2))
                        #weight vector not taken care of in fs_rem calculation, and it is not required                   
                        #print("i = %d, flip score of remaining attack is %.2f" %(i+1,fs_rem)) 
                    break
    return param_list

#Adaptive krum attack
def adaptive_krum(device, lr, v, old_direction, f, fs_min, fs_max):

    if (f==0):
        return v
    e = 0.0001/len(v[0])
    direction = torch.sign(torch.sum(v, axis=0))
    ben_grads = torch.mean(v, axis=0).to(device)
    topk = torch.argsort(torch.abs(ben_grads).reshape(-1))

    l_max = lambda_max(device, v, f)
    l = find_lambda(device, l_max, v, direction, len(v), f)

    target_attack = -(direction*l)
    direction = torch.sign(target_attack)
    flip = torch.sign(direction*(direction-old_direction.reshape(-1)))
    flip_score = torch.sum(flip*(target_attack**2))
    if (l>0):
        v[0] = -(direction*l)
        for i in range(1, f):
            noise = torch.FloatTensor(1, len(v[0])).uniform_(-e, e).to(device)
            v[i] = v[0] + noise
        return v
    if (flip_score < fs_max): return v
    else:
        step = 5
        for perc in range (0, 100, step):   
            start_idx = round((perc/100)*len(topk))
            end_idx = round(((perc+step)/100)*len(topk))
            target_attack[topk[start_idx:end_idx]] = ben_grads[topk[start_idx:end_idx]]
            direction = torch.sign(target_attack)
            flip = torch.sign(direction*(direction-old_direction.reshape(-1)))
            flip_score = torch.sum(flip*(target_attack**2))

            if (flip_score < fs_max): 
                for i in range(f):
                    noise = torch.FloatTensor(1, len(target_attack)).uniform_(-e, e).to(device)
                    v[i] = target_attack + noise
                return v


        
    

