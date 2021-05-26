from __future__ import print_function
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon, init
import numpy as np
import random
import argparse
import byzantine
import sys
import os
import pdb
from mxnet.gluon import nn

from mxnet.gluon.block import HybridBlock
from numpy import genfromtxt

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from mxnet.gluon.data.vision import transforms

np.warnings.filterwarnings('ignore')

from gluoncv.model_zoo import get_model

def get_lr(max_lr, epoch, num_epochs):

    mu = num_epochs/4
    sigma = num_epochs/4
    epoch = epoch + 1
    if (epoch < num_epochs/4):
        return max_lr*(1-np.exp(-25*(epoch/num_epochs)))
    else:
        #return 0.1*(np.exp(-7.5*(epoch - num_epochs/4)/num_epochs))
        return max_lr*np.exp(-0.5*(((epoch-mu)/sigma)**2))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='cifar10', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.5)
    parser.add_argument("--net", help="net", default='resnet20', type=str, choices=['mlr', 'dnn10', 'dnn2', 'resnet20'])
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.003, type=float)
    parser.add_argument("--nworkers", help="# workers", default=3, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=2, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=-1, type=int)
    parser.add_argument("--seed", help="seed", default=42, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=1, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='full_trim', type=str,
                        choices=['no', 'partial_trim', 'full_trim', 'full_krum'])
    parser.add_argument("--aggregation", help="aggregation rule", default='foolsgold', type=str)
    parser.add_argument("--cmax", help="FLAIR's notion of c_max", default=1, type=int)
    parser.add_argument("--utrg", help="Trigger threshold", default=1.0, type=float)
    parser.add_argument("--udet", help="Detection threshold", default=0.50, type=float)
    parser.add_argument("--urem", help="Removal threshold", default=3, type=int)
    parser.add_argument("--filename", help="Directory name", default='', type=str)
    parser.add_argument("--decay", help="Decay rate", default=2.0, type=float)
    return parser.parse_args()

def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

# Blocks
class BasicBlockV1(HybridBlock):
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(_conv3x3(channels, stride, in_channels))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels, 1, channels))
        self.body.add(nn.BatchNorm())
        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x
    
# Nets
class ResNetV1(HybridBlock):
    def __init__(self, block, layers, channels, classes=10, thumbnail=False, **kwargs):
        super(ResNetV1, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 3))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False,
                                            in_channels=3))
                self.features.add(nn.BatchNorm())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=channels[i]))

            self.classifier = nn.HybridSequential(prefix='')
            self.classifier.add(nn.GlobalAvgPool2D())
            self.classifier.add(nn.Dense(classes, in_units=channels[-1]))

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.classifier(x)

        return x
    
resnet_spec = {20: ('basic_block', [2, 2, 2, 3], [64, 64, 128, 256, 512])}    
resnet_net_versions = [ResNetV1]
resnet_block_versions = [{'basic_block': BasicBlockV1}]

def get_resnet(version, num_layers, ctx, pretrained=False, root='~/.mxnet/models', **kwargs):
    block_type, layers, channels = resnet_spec[num_layers]
    resnet_class = resnet_net_versions[version-1]
    block_class = resnet_block_versions[version-1][block_type]
    net = resnet_class(block_class, layers, channels, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_params(get_model_file('resnet%d_v%d'%(num_layers, version),
                                       root=root), ctx=ctx)
    return net

def resnet20_v1(ctx, **kwargs):
    return get_resnet(1, 20, ctx, **kwargs)

def main(args):
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    with ctx:

        batch_size = args.batch_size

        if ((args.dataset == 'mnist') or (args.dataset == 'fmnist')):
            num_inputs = 28 * 28
            num_outputs = 10
        elif args.dataset == 'chmnist':
            num_inputs = 64*64
            num_outputs = 8
        elif args.dataset == 'bcw':
            num_inputs = 30
            num_outputs = 2
        elif args.dataset == 'cifar10':
            num_inputs = 32*32*3
            num_outputs = 10
        else:
            sys.exit('Not Implemented Dataset!')
        
        
        #################################################
        # Multiclass Logistic Regression
        MLR = gluon.nn.Sequential()
        with MLR.name_scope():
            MLR.add(gluon.nn.Dense(num_outputs))
            
        ########################################################################################################################
        def evaluate_accuracy(data_iterator, net):

            acc = mx.metric.Accuracy()
            for i, (data, label) in enumerate(data_iterator):
                if args.net == 'mlr':
                    data = data.as_in_context(ctx).reshape((-1, num_inputs))
                    label = label.as_in_context(ctx)
                elif args.net == 'dnn10' and (args.dataset == 'mnist' or args.dataset == 'fmnist'):
                    data = data.as_in_context(ctx).reshape((-1,1,28,28))
                    label = label.as_in_context(ctx)
                elif args.dataset == 'chmnist':
                    data = data.as_in_context(ctx).reshape((-1,1,64,64))
                    label = label.as_in_context(ctx)
                elif args.net == 'dnn2':
                    data = data.as_in_context(ctx).reshape((-1, 1, 1, num_inputs))
                    label = label.as_in_context(ctx)
                elif args.dataset == 'cifar10':
                    data = data.as_in_context(ctx).reshape((-1,3,32,32))
                    label = label.as_in_context(ctx)
                output = net(data)
                predictions = nd.argmax(output, axis=1)
                if args.dataset == 'chmnist':
                    predictions = predictions.reshape(-1,1)   		
                acc.update(preds=predictions, labels=label)
            return acc.get()[1]


        ########################################################################################################################
        # decide attack type
        if args.byz_type == 'partial_trim':
            # partial knowledge trim attack
            byz = byzantine.partial_trim
        elif args.byz_type == 'full_trim':
            # full knowledge trim attack
            byz = byzantine.full_trim
        elif args.byz_type == 'full_krum':
            byz = byzantine.full_krum
        elif args.byz_type == 'no':
            byz = byzantine.no_byz
        else:
            sys.exit('Not Implemented Attack!')

        # decide model architecture
        if args.net == 'mlr':
            net = MLR
            net.collect_params().initialize(mx.init.Xavier(magnitude=1.), force_reinit=True, ctx=ctx)
        elif args.net == 'dnn10':
            net = nn.Sequential()
            net.add(nn.Conv2D(channels=30, kernel_size=3, activation='relu'),
                     nn.MaxPool2D(pool_size=2, strides=2),
                     nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
                     nn.MaxPool2D(pool_size=2, strides=2),
                     nn.Flatten(),
                     nn.Dense(200, activation='relu'),
                     nn.Dense(10))
            net.collect_params().initialize(mx.init.Xavier(magnitude=1.), force_reinit=True, ctx=ctx)
        elif args.net == 'dnn2':
            net = nn.Sequential()
            net.add(nn.Conv2D(channels=30, kernel_size=1, activation='relu'),
                     nn.MaxPool2D(pool_size=1, strides=1),
                     nn.Conv2D(channels=50, kernel_size=1, activation='relu'),
                     nn.MaxPool2D(pool_size=1, strides=1),
                     nn.Flatten(),
                     nn.Dense(200, activation='relu'),
                     nn.Dense(2))
            net.initialize(init=init.Xavier(), ctx=ctx)
            #net.collect_params().initialize(mx.init.Xavier(magnitude=1.), force_reinit=True, ctx=ctx)        
        elif args.net == 'resnet20':
            net = get_model('cifar_resnet20_v1', pretrained=False, classes=8, ctx=ctx)
            net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
          
        else:
            sys.exit('Not Implemented model architecture!')


        # define loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        # set upt parameters
        num_workers = args.nworkers
        lr = args.lr
        epochs = args.nepochs
        cmax = args.cmax
        dec = args.decay
        grad_list = []
        train_acc_list = []
        
        # generate a string indicating the parameters
        paraString = str(args.byz_type) + "_" + str(args.aggregation) + "_" + str(
            args.dataset) + "_" + str(args.net) + "_lr_" + str(args.lr) + "_bias_" + str(
            args.bias) + "_m_" + str(args.nworkers) + "_c_" + str(
            args.nbyz) + "_cmax_" + str(args.cmax) + "_d_" + str(
            args.decay) + "_batch_" + str(args.batch_size) + "_epochs_" + str(args.nepochs) + "_"

        # set up seed
        seed = args.seed
        mx.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # load dataset
        if (args.dataset == 'mnist'):
            def transform(data, label):
                return data.astype(np.float32) / 255, label.astype(np.float32)
                
            test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.datasets.MNIST(train=False, transform=transform), 500, shuffle=False, last_batch='rollover') 
            train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.datasets.MNIST(train=True, transform=transform), 60000, shuffle=True, last_batch='rollover')
        
        elif (args.dataset == 'cifar10'):
            transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
            transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]) 	    
            test_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=False).transform_first(transform_test), batch_size=32, shuffle=False)
            train_data = gluon.data.DataLoader(gluon.data.vision.CIFAR10(train=True).transform_first(transform_train), batch_size=32, shuffle=True, last_batch='discard')

        elif (args.dataset == 'fmnist'):
            def transform(data, label):
                return data.astype(np.float32) / 255, label.astype(np.float32)
                
            test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.datasets.FashionMNIST(train=False, transform=transform), 500, shuffle=False, last_batch='rollover') 
            train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.datasets.FashionMNIST(train=True, transform=transform), 60000, shuffle=True, last_batch='rollover')
         
            
        elif (args.dataset == 'chmnist'):
            chdata = genfromtxt('chmnist64_shuffled.csv', delimiter=',')
            train_data_ = chdata[1:4001]
            test_data_ = chdata[4001:] 
            train_data = mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(train_data_[:, 1:-1].astype(np.float32)/255, train_data_[:, -1:].astype(np.float32)-1), 4000, shuffle=False, last_batch='rollover')
            test_data = mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(test_data_[:, 1:-1].astype(np.float32)/255, test_data_[:, -1:].astype(np.float32)-1), 1000, shuffle=True, last_batch='rollover')            
         
        elif (args.dataset == 'bcw'):
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            y = data.target
            df = (df - df.mean()) / (df.max() - df.min())  
            X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20, random_state=69)  
            train_data = mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(X_train.values.astype(np.float32), y_train.astype(np.float32)), 455, shuffle=False, last_batch='rollover')
            test_data = mx.gluon.data.DataLoader(mx.gluon.data.dataset.ArrayDataset(X_test.values.astype(np.float32), y_test.astype(np.float32)), 114, shuffle=True, last_batch='rollover')
        else:
            sys.exit('Not Implemented dataset!')
            
        # biased assignment
        bias_weight = args.bias
        other_group_size = (1-bias_weight) / (num_outputs-1)
        worker_per_group = num_workers / (num_outputs)
        
        # assign non-IID training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)] 

        counter = 0
        server_data = mx.nd.empty((100,1,28,28))
        server_label = mx.nd.empty(100)
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                if (args.dataset == 'mnist' or args.dataset == 'fmnist') and args.net == 'mlr':
                    x = x.as_in_context(ctx).reshape(-1, num_inputs)
                if (args.dataset == 'mnist' or args.dataset == 'fmnist') and args.net == 'dnn10':
                    x = x.as_in_context(ctx).reshape(-1, 1, 28, 28)
                if args.dataset == 'chmnist':
                    x = x.as_in_context(ctx).reshape(-1, 1, 64, 64)
                if args.dataset == 'bcw':
                    x = x.as_in_context(ctx).reshape(-1, 1, 1, 30)
                if args.dataset == 'cifar10':
                    x = x.as_in_context(ctx).reshape(-1, 3, 32, 32)
                y = y.as_in_context(ctx)
                
                # assign a data point to a group
                upper_bound = (y.asnumpy()) * (1-bias_weight) / (num_outputs-1) + bias_weight
                lower_bound = (y.asnumpy()) * (1-bias_weight) / (num_outputs-1)
                rd = np.random.random_sample()
                
                if rd > upper_bound:
                    worker_group = int(np.floor((rd - upper_bound) / other_group_size)+y.asnumpy()+1)
                elif rd < lower_bound:
                    worker_group = int(np.floor(rd / other_group_size))
                else:
                    worker_group = y.asnumpy()
                
                # assign a data point to a worker
                rd = np.random.random_sample()
                selected_worker = int(worker_group*worker_per_group + int(np.floor(rd*worker_per_group)))
                each_worker_data[selected_worker].append(x)
                each_worker_label[selected_worker].append(y)
                
                if (args.aggregation == 'fltrust'):
                    if (counter < 100):
                        server_data[counter] = x.reshape((1,28,28))
                        server_label[counter] = y
        # concatenate the data for each worker
        each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data] 
        each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]
        #pdb.set_trace() 
        # random shuffle the workers
        random_order = np.random.RandomState(seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
        P = 0
        if (args.net == 'mlr' and (args.dataset == 'mnist' or args.dataset == 'fmnist')):
            shape = (1, 784)
        elif (args.net == 'dnn10' and (args.dataset == 'mnist' or args.dataset == 'fmnist')):
            shape = (1, 1, 28, 28)
        elif (args.dataset == 'chmnist'):
            shape = (1, 1, 64, 64)
        elif (args.dataset == 'bcw'):
            shape = (1, 1, 1, 30)
        elif (args.dataset == 'cifar10'):
            shape = (1, 3, 32, 32)

        dummy_output = net(mx.nd.zeros(shape))
        # count the total number of parameters in the network
        for param in net.collect_params().values():
            if param.grad_req != 'null':
                P = P + len(param.grad().reshape(-1))
        #pdb.set_trace() 
        if (args.aggregation == 'EULtrim') or (args.aggregation == 'EULkrum') or (args.aggregation == 'EULmedian'):
            if args.dataset == 'mnist':
                valid_dataset = mx.gluon.data.vision.datasets.MNIST(train=False, transform=transform)
            if args.dataset == 'fmnist':
                valid_dataset = mx.gluon.data.vision.datasets.FashionMNIST(train=False, transform=transform)      
            sampled = np.random.choice(10000,100)
            valid_array = mx.gluon.data.dataset.ArrayDataset(valid_dataset[sampled[:]][0], valid_dataset[sampled[:]][1])
            valid_data = mx.gluon.data.DataLoader(valid_array, 100, shuffle=True)

            del valid_dataset
            del valid_array
        direction = mx.nd.zeros(P) #current direction of the global model
        flip_vector = np.empty(epochs) #flipscore of the global model
        local_flip_vector = np.zeros((epochs, num_workers)) #flipscore of all local models
        local_flip_new = np.zeros((epochs, num_workers))
        active = np.arange(num_workers) #used in the earlier version to know which clients aren't blacklisted yet
        blacklist = np.zeros(num_workers)
        susp = nd.zeros(num_workers) #suspicion score of all clients
        test_acc = np.empty(epochs) 
        corrected = epochs #in which epoch were cmax clients removed
        flag_corrected = 1
        max_flip = 1.0 #used for the whitebox adaptive attack
        client_list = np.ones((epochs, num_workers))*(-1) #clients chosen by EUL/FABA etc
        # begin training        
        for e in range(epochs):
            #print (lr)
            #if (e == 200): lr = lr/2
            #if (e == 400): lr = lr/2
            if (args.aggregation == 'fltrust'):
                with autograd.record():
                    output = net(server_data)
                    loss = softmax_cross_entropy(output, server_label)
                loss.backward()
                server_params = [param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null']     
            for i in range(num_workers):
                if (blacklist[i] == 0):
                    # sample a batch
                    minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=batch_size, replace=False)
                    # forward
                    with autograd.record():
                        output = net(each_worker_data[i][minibatch])
                        loss = softmax_cross_entropy(output, each_worker_label[i][minibatch])
                        # backward
                    loss.backward()
                    grad_list.append([param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null'])
            
            if cmax > 0:
                flag_corrected = 1
            susp = susp/dec
            #lr = get_lr(args.lr, e, epochs) 
            if args.aggregation == 'trim1':
                # we aggregate the gradients instead of local model weights in this demo because for the 
                # aggregation rules in our setting, it is equivalent to aggregate either of them
                _, direction, cmax, flip_count, lfs = nd_aggregation.trim1(e, grad_list, net, lr, byz, direction,
                                           active, blacklist, susp, args.nbyz, cmax, args.utrg, args.udet, args.urem) 

                flip_vector[e] = flip_count
                local_flip_vector[e] = lfs.asnumpy()
                
            if args.aggregation == 'trim':
                # we aggregate the gradients instead of local model weights in this demo because for the 
                # aggregation rules in our setting, it is equivalent to aggregate either of them
                _, direction, cmax, flip_count, lfs, lfs_new = nd_aggregation.trim(e, grad_list, net, lr, byz, direction,
                                           active, blacklist, susp, args.nbyz, cmax, args.utrg, args.udet, args.urem) 

                flip_vector[e] = flip_count
                local_flip_vector[e] = lfs.asnumpy()
                local_flip_new[e] = lfs_new.asnumpy()

            elif args.aggregation == 'fltrust':
                nd_aggregation.fltrust(e, server_params, grad_list, net, lr, byz, args.nbyz, active)

            elif args.aggregation == 'foolsgold':
                nd_aggregation.foolsgold(e, grad_list, net, lr, byz, args.nbyz, active) 
                
            elif args.aggregation == 'krum':
                _, direction, cmax, flip_count, lfs, max_flip = nd_aggregation.krum(e, grad_list, net, lr, byz,
                                           direction, active, blacklist, susp, args.nbyz, cmax, args.utrg,
                                           args.udet, args.urem, max_flip) 
                flip_vector[e] = flip_count
                local_flip_vector[e] = lfs.asnumpy()
                
            elif args.aggregation == 'median':
                _, direction, cmax, flip_count, lfs = nd_aggregation.median(e, grad_list, net, lr, byz, direction,
                                           active, blacklist, susp, args.nbyz, cmax, args.utrg, args.udet, args.urem) 
                flip_vector[e] = flip_count
                local_flip_vector[e] = lfs.asnumpy()
                
            elif args.aggregation == 'bulyan':
                _, bul_list = nd_aggregation.bulyan(e, grad_list, net, lr, byz, args.nbyz) 
                client_list[e] = bul_list

            elif args.aggregation == 'faba':
                faba_list = nd_aggregation.faba(e, grad_list, net, lr, byz, args.nbyz)
                client_list[e] = faba_list
             
            elif args.aggregation == 'EULtrim':
                _, eul_list = nd_aggregation.EULtrim(e, grad_list, net, lr, byz, valid_data, args.net, args.nbyz, args.gpu)
                client_list[e] = eul_list
                
            elif args.aggregation == 'EULkrum':
                _, eul_list = nd_aggregation.EULkrum(e, grad_list, net, lr, byz, valid_data, args.net, args.nbyz, args.gpu)
                client_list[e] = eul_list                
 
            elif args.aggregation == 'EULmedian':
                _, eul_list = nd_aggregation.EULmedian(e, grad_list, net, lr, byz, valid_data, args.net, args.nbyz, args.gpu)
                client_list[e] = eul_list
                
            else:
                sys.exit('Not Implemented aggregation!')
            
            if (cmax == 0 and flag_corrected == 1):
                corrected = e
                flag_corrected = 0
            # free memory
            del grad_list
            # reset the list
            grad_list = []
            
            # compute training accuracy every 10 iterations
            '''if (e+1) % 1 == 0:
                pdb.set_trace()
                train_accuracy = evaluate_accuracy(train_data, net)
                train_acc_list.append(train_accuracy)
                print("Epoch %02d. Train_acc %0.4f" % (e, train_accuracy))
            

            # save the training accuracy every 100 iterations
            if (e+1) % 1 == 0:
                if (args.dataset == 'mnist' and args.net == 'mlr'):
                    if not os.path.exists('out_mnist_mlr/'):
                        os.mkdir('out_mnist_mlr/')
                    np.savetxt('out_mnist_mlr/' + paraString, train_acc_list, fmt='%.4f')
                elif (args.dataset == 'mnist' and args.net == 'cnn'):
                    if not os.path.exists('out_mnist_cnn/'):
                        os.mkdir('out_mnist_cnn/')
                    np.savetxt('out_mnist_cnn/' + paraString, train_acc_list, fmt='%.4f')
            '''        
            # compute the final testing accuracy
            #if (e+1) == args.nepochs:
            test_accuracy = evaluate_accuracy(test_data, net)
            test_acc[e] = test_accuracy
            print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
        
        filename = args.filename
        myString = args.aggregation+'_'+args.byz_type+'_'+args.net+'_'+args.dataset+'_'+str(args.utrg)+'_'+str(args.udet)+'_'+str(args.urem)+'_'+filename+'_'
        if not os.path.exists('Outputs/'):
            os.mkdir('Outputs/')
        #np.save('Outputs/'+paraString+'Flip_old.npy', local_flip_new )
        np.save('Outputs/'+paraString+'Test_acc.npy', test_acc)
        #np.save('Outputs/'+paraString+'FLip_local_old.npy', local_flip_vector)
        #np.save('Outputs/'+paraString+'Reputation_old.npy', susp.asnumpy())
        net.save_parameters('Outputs/'+paraString+'net.params')
        ones = nd.ones(num_workers)
        zeros = nd.zeros(num_workers)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        if (args.aggregation == 'krum' or args.aggregation == 'trim' or args.aggregation == 'median'):
            for i in range(epochs):
                sflip = np.argsort(local_flip_vector[i])
                c_removed = len(np.where(local_flip_vector[i] == 0)[0])
                cmax_then = args.cmax - c_removed
                if (cmax_then > 0):
                    tp = tp + len(np.where(sflip[-cmax_then:]<args.nbyz)[0])
                    fp = fp + len(np.where(sflip[-cmax_then:]>=args.nbyz)[0])
                    tn = tn + len(np.where(sflip[c_removed:-cmax_then]>=args.nbyz)[0])
                    fn = fn + len(np.where(sflip[c_removed:-cmax_then]<args.nbyz)[0])
                else:
                    tn = tn + len(np.where(sflip[c_removed:]>=args.nbyz)[0])
                    fn = fn + len(np.where(sflip[c_removed:]<args.nbyz)[0])                

        if (args.aggregation == 'bulyan' or args.aggregation == 'faba' or args.aggregation == 'EULtrim' or args.aggregation == 'EULkrum' or args.aggregation == 'EULmedian'):
            for i in range (epochs):
                positives = len(np.where(client_list[i] == -1)[0])
                negatives = num_workers - positives
                tn = tn + len(np.where(client_list[i, :negatives] >= args.nbyz)[0])
                fn = fn + len(np.where(client_list[i, :negatives] < args.nbyz)[0])
                tp = tp + args.nbyz - len(np.where(client_list[i, :negatives] < args.nbyz)[0])
                fp = fp + num_workers - args.nbyz - len(np.where(client_list[i, :negatives] >= args.nbyz)[0])

        print (tp, fp, tn, fn, corrected)     
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
