import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
from copy import deepcopy
import time
#import FLAIR
import pdb

def fltrust(epoch, server_params, gradients, net, lr, byz, f, active):

    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    server_params = nd.concat(*[xx.reshape((-1,1)) for xx in server_params], dim=0)
    num_workers = len(param_list)
    param_list = byz(epoch, param_list, net, f, lr, active)
    server_norm = mx.nd.norm(server_params)
    ts = mx.nd.zeros(num_workers)

    for i in range(num_workers):
        client_norm = mx.nd.norm(param_list[i])
        cos = ((mx.nd.sum(server_params.reshape(-1)*param_list[i].reshape(-1)))/(server_norm*client_norm)).asscalar()
        ts[i] = max(cos, 0)
        param_list[i] = (server_norm/mx.nd.norm(param_list[i]))*param_list[i]*ts[i]

    global_params = sum(param_list)/mx.nd.sum(ts)
    del param_list
    idx = 0   
    for j, (param) in enumerate(net.collect_params().values()): 
        if param.grad_req == 'null':    
            continue   
        param.set_data(param.data() - lr * global_params[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size 
    return  


def foolsgold(epoch, gradients, net, lr, byz, f, active):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    num_workers = len(param_list)
    param_list = byz(epoch, param_list, net, f, lr, active)
    cs = mx.nd.zeros((num_workers, num_workers))
    for i in range (num_workers):
        for j in range (i):
            aa = param_list[i]/mx.nd.norm(param_list[i])
            bb = param_list[j]/mx.nd.norm(param_list[j])
            cs[i,j] = mx.nd.sum((aa.reshape(-1))*(bb.reshape(-1))).asscalar()
            cs[j,i] = cs[i,j]
            del aa, bb
    
    v = mx.nd.zeros(num_workers)      
    for i in range (num_workers):
        v[i] = mx.nd.max(cs[i])
      
    alpha = mx.nd.zeros(num_workers)
    for i in range (num_workers):
        for j in range (num_workers):
            if (v[j] > v[i]):
                cs[i,j] = cs[i,j]*v[i]/v[j]
        alpha[i] = 1 - mx.nd.max(cs[i])
    
    ones = mx.nd.ones(num_workers)
    zeros = mx.nd.zeros(num_workers)
    nines = mx.nd.ones(num_workers)*0.99

    alpha = mx.nd.where((alpha > 1), ones, alpha)
    alpha = mx.nd.where((alpha < 0), zeros, alpha)
    alpha = alpha/(mx.nd.max(alpha))
    alpha = mx.nd.where((alpha == 1), nines, alpha)
    alpha = mx.nd.log(alpha/(1-alpha)) + 0.5
    alpha = mx.nd.where((mx.nd.contrib.isinf(alpha) + mx.nd.contrib.isnan(alpha) + (alpha > 1) > 0), ones, alpha)
    alpha = mx.nd.where((alpha < 0), zeros, alpha)
    alpha = alpha/mx.nd.sum(alpha)
    print (alpha)
    matrix = nd.transpose(nd.transpose(nd.concat(*[ii for ii in param_list], dim=1)))
    global_params = nd.linalg.gemm2(matrix, alpha.reshape(-1,1))   
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * global_params[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    
    return

# flair
def trim(epoch, gradients, net, lr, byz, old_direction, active, blacklist, susp, f = 0,
         cmax = 0, utrg = 0.0, udet = 0.50, urem = 3):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, f, lr, active)
    flip_local = nd.zeros(len(param_list))    
    flip_new = nd.zeros(len(param_list)) 
    penalty = 1.0 - cmax/len(param_list)
    reward = 1.0 - penalty
    
    for i in range (len(param_list)):
        direction = nd.sign(param_list[i])
        flip_local[i] = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
        #flip = nd.sign(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))
        #flip_new[i] = nd.sum(flip*(param_list[i].reshape(-1)**2))

        #flip[param_list[i]<0.0001] = 0
        #flip_new[i] = nd.sum(flip).asscalar()
    #argsorted = nd.argsort(flip_local) 
    argsorted = nd.argsort(flip_local)
    if (cmax > 0):
        susp[argsorted[:-cmax]] = susp[argsorted[:-cmax]] + reward
        susp[argsorted[-cmax:]] = susp[argsorted[-cmax:]] - penalty
    argsorted = nd.argsort(susp)
    weights = nd.exp(susp) / nd.sum(nd.exp(susp))
    matrix = nd.transpose(nd.transpose(nd.concat(*[ii for ii in param_list], dim=1)))
    trim_nd = nd.linalg.gemm2(matrix, weights.reshape(-1,1))   
    #pdb.set_trace()
    #print (flip_new, weights)
    #print (nd.mean(flip_local[:cmax]), nd.mean(flip_new[:cmax]), nd.mean(flip_local[cmax:]), nd.mean(flip_new[cmax:]))   
    '''new_list = []
    argsorted = nd.argsort(susp) 
    for i in range(len(param_list)-cmax):
        new_list.append(param_list[int(argsorted[i].asscalar())])
    
    sorted_array = nd.sort(nd.concat(*new_list, dim=1), axis=-1)
    trim_nd = nd.mean(sorted_array, axis=-1, keepdims=1)'''
    global_direction = nd.sign(trim_nd) 
    gfs = 0.5*(mx.nd.sum(global_direction.reshape(-1)*(global_direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()

    '''if (utrg > 0):
        sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
        n = len(param_list)
        m = n - f*2
        trim_nd = nd.mean(sorted_array[:, f:(f+m)], axis=-1, keepdims=1)    
        direction = nd.sign(trim_nd) 
        gfs = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
    
    if ((utrg>0 and gfs>=utrg*len(param_list[0])) or (utrg == 0)):
        flip_score = mx.nd.zeros(len(param_list))
        rem = []
        for i in range (len(param_list)):
            direction = nd.sign(param_list[i])
            flip_score[i] = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
            flip_local[active[i]] = flip_score[i].asscalar()
        argsorted = nd.argsort(flip_score) 
        new_list = []
        for i in range(len(param_list)-cmax):
            new_list.append(param_list[int(argsorted[i].asscalar())])
            
        n = len(new_list)
        f = 0
        m = n - f*2
        sorted_array = nd.sort(nd.concat(*new_list, dim=1), axis=-1)
        trim_nd = nd.mean(sorted_array[:, f:(f+m)], axis=-1, keepdims=1) 
        
        for i in range(len(new_list), len(param_list)):
            index = int(argsorted[i].asscalar())
            if (flip_score[index] >= udet*len(param_list[0])):
                susp[active[index]] = susp[active[index]] + 1
                if (susp[active[index]] >= urem):
                    blacklist[active[index]] = 1
                    rem.append(active[index])
        active = removearr(active, sorted(rem), len(param_list))             
             
        direction = nd.sign(trim_nd)
        gfs = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
        cmax = cmax - len(rem)      '''  

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    return trim_nd, direction, cmax, gfs, flip_local, flip_new

def krum(epoch, gradients, net, lr, byz, old_direction, active, blacklist, susp, f = 0, 
         cmax = 0, utrg = 0, udet = 0.50, urem = 3, max_flip=1.0):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, f, lr, active, max_flip)
  
    flip_local = nd.zeros(len(param_list))    
    penalty = 1.0 - cmax/len(param_list)
    reward = 1.0 - penalty
    
    for i in range (len(param_list)):
        direction = nd.sign(param_list[i])
        flip_local[i] = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
    argsorted = nd.argsort(flip_local) 
    susp[argsorted[:-cmax]] = susp[argsorted[:-cmax]] - reward
    susp[argsorted[-cmax:]] = susp[argsorted[-cmax:]] + penalty
    
    new_list = []
    argsorted = nd.argsort(susp) 
    for i in range(len(param_list)-cmax):
        new_list.append(param_list[int(argsorted[i].asscalar())])

    k = len(new_list) - 0 - 2
    dist = mx.nd.zeros((len(new_list),len(new_list)))
    for i in range (0, len(new_list)):
        for j in range(0, i):
            dist[i][j] = nd.norm(new_list[i] - new_list[j])
            dist[j][i] = dist[i][j]     

    sorted_dist = mx.nd.sort(dist)
    sum_dist = mx.nd.sum(sorted_dist[:,:k+1], axis=1)
    model_selected = argsorted[mx.nd.argmin(sum_dist).asscalar().astype(int)].asscalar().astype(int)   
    global_direction = nd.sign(param_list[model_selected])            
    gfs = 0.5*(mx.nd.sum(global_direction.reshape(-1)*(global_direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()            
    '''if (utrg > 0):
        k = len(param_list) - f - 2
        dist = mx.nd.zeros((len(param_list),len(param_list)))
        for i in range (0, len(param_list)):
            for j in range(0, i):
                dist[i][j] = nd.norm(param_list[i] - param_list[j])
                dist[j][i] = dist[i][j]    
            
        sorted_dist = mx.nd.sort(dist)
        sum_dist = mx.nd.sum(sorted_dist[:,:k+1], axis=1)
        model_selected = mx.nd.argmin(sum_dist).asscalar().astype(int)   
        direction = nd.sign(param_list[model_selected])
        gfs = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()

        for i in range (len(param_list)):
            direction = nd.sign(param_list[i])
            flip_score = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
            flip_local[active[i]] = flip_score
        
    if ((utrg>0 and gfs>=utrg*len(param_list[0])) or (utrg == 0)):
        flip_score = mx.nd.zeros(len(param_list))
        rem = []
        for i in range (len(param_list)):
            direction = nd.sign(param_list[i])
            flip_score[i] = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
            flip_local[active[i]] = flip_score[i].asscalar()
        argsorted = nd.argsort(flip_score) 
        new_list = []
        for i in range(len(param_list)-cmax):
            new_list.append(param_list[int(argsorted[i].asscalar())])
            
        k = len(new_list) - 0 - 2
        dist = mx.nd.zeros((len(new_list),len(new_list)))
        for i in range (0, len(new_list)):
            for j in range(0, i):
                dist[i][j] = nd.norm(new_list[i] - new_list[j])
                dist[j][i] = dist[i][j]    
                
        for i in range(len(new_list), len(param_list)):
            index = int(argsorted[i].asscalar())
            if (flip_score[index] >= udet*len(param_list[0])):
                susp[active[index]] = susp[active[index]] + 1
                if (susp[active[index]] >= urem):
                    blacklist[active[index]] = 1
                    rem.append(active[index])
        active = removearr(active, sorted(rem), len(param_list))             
            
        sorted_dist = mx.nd.sort(dist)
        sum_dist = mx.nd.sum(sorted_dist[:,:k+1], axis=1)
        model_selected = argsorted[mx.nd.argmin(sum_dist).asscalar().astype(int)].asscalar().astype(int)   
        direction = nd.sign(param_list[model_selected])
        gfs = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
        cmax = cmax - len(rem)    '''
                    
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * param_list[model_selected][idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size  

    return model_selected, direction, cmax, gfs, flip_local, 1.0 #flip_score[len(param_list)-cmax-1].asscalar()/len(param_list[0])

def removearr(clients, arr, m):
    kk = 0
    for i in range(m):
        if (i>0 and clients[i] <= clients[i-1]):
            if (i+kk < len(clients)):
                clients[i] = clients[i+kk]
            else:
                clients[i] = -1
        elif (i>0 and clients[i-1]==-1):
            clients[i] = -1
        #print (i, clients)
        while (kk<len(arr) and clients[i] == arr[kk]):
            #print ("Incrementing kk")
            kk = kk + 1
            if (i+kk < len(clients)):
                clients[i] = clients[i+kk]
            else:
                clients[i] = -1
            #print (clients)
    return clients

def median(epoch, gradients, net, lr, byz, old_direction, active, blacklist, susp, f = 0,
         cmax = 0, utrg = 0.0, udet = 0.50, urem = 3):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, f, lr, active)
  
    flip_local = nd.zeros(len(param_list))    
    penalty = 1.0 - cmax/len(param_list)
    reward = 1.0 - penalty
    
    for i in range (len(param_list)):
        direction = nd.sign(param_list[i])
        flip_local[i] = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
    argsorted = nd.argsort(flip_local) 
    susp[argsorted[:-cmax]] = susp[argsorted[:-cmax]] - reward
    susp[argsorted[-cmax:]] = susp[argsorted[-cmax:]] + penalty
    
    new_list = []
    argsorted = nd.argsort(susp) 
    for i in range(len(param_list)-cmax):
        new_list.append(param_list[int(argsorted[i].asscalar())])
    
    sorted_array = nd.sort(nd.concat(*new_list, dim=1), axis=-1)
    if (len(new_list)%2 == 1):
        trim_nd = sorted_array[:, int(len(new_list)/2)]
    else:
        trim_nd = (sorted_array[:, int(len(new_list)/2)-1] + sorted_array[:, int(len(new_list)/2)])/2    
    global_direction = nd.sign(trim_nd) 
    gfs = 0.5*(mx.nd.sum(global_direction.reshape(-1)*(global_direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
    
    '''if (utrg > 0):
        sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
        n = len(param_list)
        m = n - f*2
        if (len(param_list)%2 == 1):
            trim_nd = sorted_array[:, int(len(param_list)/2)]
        else:
            trim_nd = (sorted_array[:, int(len(param_list)/2)-1] + sorted_array[:, int(len(param_list)/2)])/2    
        direction = nd.sign(trim_nd) 
        gfs = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
    
    if ((utrg>0 and gfs>=utrg*len(param_list[0])) or (utrg == 0)):
        flip_score = mx.nd.zeros(len(param_list))
        rem = []
        for i in range (len(param_list)):
            direction = nd.sign(param_list[i])
            flip_score[i] = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
            flip_local[active[i]] = flip_score[i].asscalar()
        argsorted = nd.argsort(flip_score) 
        new_list = []
        for i in range(len(param_list)-cmax):
            new_list.append(param_list[int(argsorted[i].asscalar())])
            
        n = len(new_list)
        f = 0
        sorted_array = nd.sort(nd.concat(*new_list, dim=1), axis=-1)
        if (len(new_list)%2 == 1):
            trim_nd = sorted_array[:, int(len(new_list)/2)]
        else:
            trim_nd = (sorted_array[:, int(len(new_list)/2)-1] + sorted_array[:, int(len(new_list)/2)])/2
        
        for i in range(len(new_list), len(param_list)):
            index = int(argsorted[i].asscalar())
            if (flip_score[index] >= udet*len(param_list[0])):
                susp[active[index]] = susp[active[index]] + 1
                if (susp[active[index]] >= urem):
                    blacklist[active[index]] = 1
                    rem.append(active[index])
        active = removearr(active, sorted(rem), len(param_list))             
             
        direction = nd.sign(trim_nd)
        gfs = 0.5*(mx.nd.sum(direction.reshape(-1)*(direction.reshape(-1)-old_direction.reshape(-1)))).asscalar()
        cmax = cmax - len(rem)     '''   

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size
    return trim_nd, direction, cmax, gfs, flip_local

def bulyan(epoch, gradients, net, lr, byz, f = 0):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, f, lr, np.arange(len(param_list)))
    
    k = len(param_list) - f - 2
    dist = mx.nd.zeros((len(param_list),len(param_list)))
    for i in range (0, len(param_list)):
        for j in range(0, i):
            dist[i][j] = nd.norm(param_list[i] - param_list[j])
            dist[j][i] = dist[i][j]    
            
    sorted_dist = mx.nd.sort(dist)
    sum_dist = mx.nd.sum(sorted_dist[:,:k+1], axis=1)
    bulyan_list = []
    bul_client_list = np.ones(len(param_list))*(-1)
    for i in range(len(param_list)-2*f):
        chosen = int(nd.argmin(sum_dist).asscalar())
        sum_dist[chosen] = 10**8
        bul_client_list[i] = chosen
        bulyan_list.append(param_list[chosen])
        for j in  range (len(sum_dist)):
            sum_dist[j] = sum_dist[j] - dist[j][chosen]
    sorted_array = nd.sort(nd.concat(*bulyan_list, dim=1), axis=-1)
    trim_nd = nd.mean(sorted_array[:, f:(len(bulyan_list)-f)], axis=-1, keepdims=1)     
                    
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size  
    return trim_nd, bul_client_list  

def faba(epoch, gradients, net, lr, byz, f = 0):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, f, lr, np.arange(len(param_list)))  
    faba_client_list = np.arange(len(param_list))
    dist = np.zeros(len(param_list))
    G0 = nd.mean(nd.concat(*param_list, dim=1), axis=-1, keepdims=1)  
    for i in range (f):      
        for j in range (len(param_list)):
            dist[j] = (nd.norm(G0 - param_list[j])*(faba_client_list[j]>=0)).asscalar()
        client = int(np.argmax(dist))
        faba_client_list[client] = -1
        dist[client] = 0
        G0 = (G0*(len(param_list)-i) - param_list[client])/(len(param_list)-i-1)

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * G0[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size    
        
    del param_list
    del dist
    del G0
    return -np.sort(-faba_client_list)

def evaluate_accuracy(data_iterator, net, net_name, gpu = -1):

    if gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(gpu)    
    acc = mx.metric.Accuracy()
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    loss = 0
    for i, (data, label) in enumerate(data_iterator):
        if net_name == 'mlr':
            data = data.as_in_context(ctx).reshape((-1, 784))
            label = label.as_in_context(ctx)
        elif net_name == 'dnn10':
            data = data.as_in_context(ctx).reshape((-1,1,28,28))
            label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
        loss = loss + nd.sum(softmax_cross_entropy(output, label)).asscalar()
                
    return acc.get()[1], loss 

def EULtrim(epoch, gradients, net, lr, byz, test_data, net_name, f = 0, gpu = -1):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, f, lr, np.arange(len(param_list))) 
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    trim_nd = nd.mean(sorted_array[:, f:-f], axis=-1, keepdims=1)  
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size  
        
    accuracy_all, loss_all = evaluate_accuracy(test_data, net, net_name, gpu)    
    prev_trim = deepcopy(trim_nd)    
    
    accuracy_np = np.zeros(len(param_list))
    loss_np = np.zeros(len(param_list))
    for i in range (len(param_list)):
        param_eul = deepcopy(param_list)
        param_eul[i] = (mx.nd.ones(len(param_list[0]))*(-10000)).reshape((-1,1))
        sorted_array = nd.sort(nd.concat(*param_eul, dim=1), axis=-1)
        trim_nd = nd.mean(sorted_array[:, f+1:-f], axis=-1, keepdims=1)  

        idx = 0
        for j, (param) in enumerate(net.collect_params().values()):
            if param.grad_req == 'null':
                continue
            param.set_data(param.data() - lr * (trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape)) + lr*(prev_trim[idx:(idx+param.data().size)].reshape(param.data().shape)))
            idx += param.data().size     
        accuracy_eul, loss_eul = evaluate_accuracy(test_data, net, net_name, gpu)
        accuracy_np[i] = accuracy_all - accuracy_eul
        loss_np[i] = loss_all - loss_eul
        prev_trim = deepcopy(trim_nd)    
    
    intersection = np.intersect1d(np.argsort(accuracy_np)[:-f], np.argsort(loss_np)[f:])    
    eul_client_list = np.ones(len(param_list))*(-1)
    eul_client_list[:len(intersection)] = intersection
    removed = np.setdiff1d(np.arange(len(param_list)), intersection)

    for i in range(len(removed)):
        param_list[int(removed[i])] = (mx.nd.ones(len(param_list[0]))*(-10000)).reshape((-1,1))
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    trim_nd = nd.mean(sorted_array[:, f+len(removed):-f], axis=-1, keepdims=1)  
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() + lr * prev_trim[idx:(idx+param.data().size)].reshape(param.data().shape) - lr * (trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape)))
        idx += param.data().size      

    del trim_nd
    del prev_trim
    del param_list
    del param_eul
        
    return net, eul_client_list
        
def EULkrum(epoch, gradients, net, lr, byz, test_data, net_name, f = 0, gpu = -1):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, f, lr, np.arange(len(param_list)))  
    k = len(param_list) - f - 2
    dist = mx.nd.zeros((len(param_list), len(param_list)))
    for i in range (len(param_list)):
        for j in range (i):
            dist[i][j] = nd.norm(param_list[i] - param_list[j])
            dist[j][i] = dist[i][j] 
            
    model_selected = mx.nd.argmin(mx.nd.sum(mx.nd.sort(dist)[:, :k+1], axis=1)).asscalar().astype(int)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * (param_list[model_selected][idx:(idx+param.data().size)].reshape(param.data().shape))) 
        idx += param.data().size     
    accuracy_all, loss_all = evaluate_accuracy(test_data, net, net_name, gpu) 
    
    accuracy_np = np.zeros(len(param_list))
    loss_np = np.zeros(len(param_list))
    
    for i in range (len(param_list)):
        prev_model = model_selected
        dist_eul = dist.copy()
        dist_eul[i] = np.nan
        dist_eul[:,i] = np.nan
        model_selected = mx.nd.argmin(mx.nd.sum(mx.nd.sort(dist_eul)[:, :k+1], axis=1)).asscalar().astype(int)
        idx = 0
        for j, (param) in enumerate(net.collect_params().values()):
            if param.grad_req == 'null':
                continue
            param.set_data(param.data() + lr * (param_list[prev_model][idx:(idx+param.data().size)].reshape(param.data().shape)) - lr*param_list[model_selected][idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size     
        accuracy_eul, loss_eul = evaluate_accuracy(test_data, net, net_name, gpu)
        accuracy_np[i] = accuracy_all - accuracy_eul
        loss_np[i] = loss_all - loss_eul
    prev_model = model_selected
    intersection = np.intersect1d(np.argsort(accuracy_np)[:-f], np.argsort(loss_np)[f:])    
    eul_client_list = np.ones(len(param_list))*(-1)
    eul_client_list[:len(intersection)] = intersection
    removed = np.setdiff1d(np.arange(len(param_list)), intersection)
    del dist_eul
    
    k = len(intersection) - f - 2   
    for i in range(len(removed)):
        dist[int(removed[i])] = 100000
        dist[:,int(removed[i])] = 100000

    model_selected = mx.nd.argmin(mx.nd.sum(mx.nd.sort(dist)[:, :k+1], axis=1)).asscalar().astype(int)      
    del dist
    del removed
    del intersection

    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() + lr * (param_list[prev_model][idx:(idx+param.data().size)].reshape(param.data().shape))- lr * param_list[model_selected][idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size 
        
    return net, eul_client_list   

def EULmedian(epoch, gradients, net, lr, byz, test_data, net_name, f = 0, gpu = -1):
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    param_list = byz(epoch, param_list, net, f, lr, np.arange(len(param_list))) 
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    if (len(param_list)%2 == 1):
        trim_nd = sorted_array[:, int(len(param_list)/2)]
    else:
        trim_nd = (sorted_array[:, int(len(param_list)/2)-1] + sorted_array[:, int(len(param_list)/2)])/2  
  
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() - lr * trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size  
        
    accuracy_all, loss_all = evaluate_accuracy(test_data, net, net_name, gpu)    
    prev_trim = deepcopy(trim_nd)    
    
    accuracy_np = np.zeros(len(param_list))
    loss_np = np.zeros(len(param_list))
    for i in range (len(param_list)):
        param_eul = deepcopy(param_list)
        param_eul[i] = (mx.nd.ones(len(param_list[0]))*(-10000)).reshape((-1,1))
        sorted_array = nd.sort(nd.concat(*param_eul, dim=1), axis=-1)
        if (len(param_eul)%2 == 1):
            trim_nd = sorted_array[:, int(len(param_eul)/2)]
        else:
            trim_nd = (sorted_array[:, int(len(param_eul)/2)-1] + sorted_array[:, int(len(param_eul)/2)])/2      
        
        idx = 0
        for j, (param) in enumerate(net.collect_params().values()):
            if param.grad_req == 'null':
                continue
            param.set_data(param.data() - lr * (trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape)) + lr*(prev_trim[idx:(idx+param.data().size)].reshape(param.data().shape)))
            idx += param.data().size     
        accuracy_eul, loss_eul = evaluate_accuracy(test_data, net, net_name, gpu)
        accuracy_np[i] = accuracy_all - accuracy_eul
        loss_np[i] = loss_all - loss_eul
        prev_trim = deepcopy(trim_nd)    
    
    intersection = np.intersect1d(np.argsort(accuracy_np)[:-f], np.argsort(loss_np)[f:])    
    eul_client_list = np.ones(len(param_list))*(-1)
    eul_client_list[:len(intersection)] = intersection
    removed = np.setdiff1d(np.arange(len(param_list)), intersection)

    for i in range(len(removed)):
        param_list[int(removed[i])] = (mx.nd.ones(len(param_list[0]))*(-10000)).reshape((-1,1))
    sorted_array = nd.sort(nd.concat(*param_list, dim=1), axis=-1)
    if (len(param_list)%2 == 1):
        trim_nd = sorted_array[:, int(len(param_list)/2)]
    else:
        trim_nd = (sorted_array[:, int(len(param_list)/2)-1] + sorted_array[:, int(len(param_list)/2)])/2      
    
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req == 'null':
            continue
        param.set_data(param.data() + lr * prev_trim[idx:(idx+param.data().size)].reshape(param.data().shape) - lr * (trim_nd[idx:(idx+param.data().size)].reshape(param.data().shape)))
        idx += param.data().size      

    del trim_nd
    del prev_trim
    del param_list
    del param_eul
        
    return net, eul_client_list    
        
        
        
        
    
    
   
