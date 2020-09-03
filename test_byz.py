from __future__ import print_function
import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import random
import argparse
import byzantine
import sys
import os
import pdb

np.warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='mnist', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.5)
    parser.add_argument("--net", help="net", default='mlr', type=str, choices=['mlr'])
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.003, type=float)
    parser.add_argument("--nworkers", help="# workers", default=10, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=10, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=-1, type=int)
    parser.add_argument("--seed", help="seed", default=42, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=3, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='full_krum', type=str,
                        choices=['no', 'partial_trim', 'full_trim', 'full_krum'])
    parser.add_argument("--aggregation", help="aggregation rule", default='krum', type=str)
    return parser.parse_args()


def main(args):
    if args.gpu == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu)

    with ctx:

        batch_size = args.batch_size

        if args.dataset == 'mnist':
            num_inputs = 28 * 28
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
                else:
                    data = data.as_in_context(ctx)
                    label = label.as_in_context(ctx)
                output = net(data)
                predictions = nd.argmax(output, axis=1)
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
        else:
            sys.exit('Not Implemented model architecture!')


        # define loss
        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

        # set upt parameters
        num_workers = args.nworkers
        lr = args.lr
        epochs = args.nepochs
        grad_list = []
        train_acc_list = []
        
        # generate a string indicating the parameters
        paraString = str(args.dataset) + "+bias " + str(args.bias) + "+net " + str(
            args.net) + "+nepochs " + str(args.nepochs) + "+lr " + str(
            args.lr) + "+batch_size " + str(args.batch_size) + "+nworkers " + str(
            args.nworkers) + "+nbyz " + str(args.nbyz) + "+byz_type " + str(
            args.byz_type) + "+aggregation " + str(args.aggregation) + ".txt"

        # set up seed
        seed = args.seed
        mx.random.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # load dataset
        if (args.dataset == 'mnist' and args.net == 'mlr'):
            def transform(data, label):
                return data.astype(np.float32) / 255, label.astype(np.float32)
                
            test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.datasets.MNIST(train=False, transform=transform), 500, shuffle=False, last_batch='rollover') 
            train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.datasets.MNIST(train=True, transform=transform), 60000, shuffle=True, last_batch='rollover')
            
        else:
            sys.exit('Not Implemented dataset!')
            
        # biased assignment
        bias_weight = args.bias
        other_group_size = (1-bias_weight) / 9.
        worker_per_group = num_workers / 10

        # assign non-IID training data to each worker
        each_worker_data = [[] for _ in range(num_workers)]
        each_worker_label = [[] for _ in range(num_workers)]    
        for _, (data, label) in enumerate(train_data):
            for (x, y) in zip(data, label):
                if args.dataset == 'mnist' and args.net == 'mlr':
                    x = x.as_in_context(ctx).reshape(-1, num_inputs)
                y = y.as_in_context(ctx)
                
                # assign a data point to a group
                upper_bound = (y.asnumpy()) * (1-bias_weight) / 9. + bias_weight
                lower_bound = (y.asnumpy()) * (1-bias_weight) / 9.
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
        
        # concatenate the data for each worker
        each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data] 
        each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

        # random shuffle the workers
        random_order = np.random.RandomState(seed=seed).permutation(num_workers)
        each_worker_data = [each_worker_data[i] for i in random_order]
        each_worker_label = [each_worker_label[i] for i in random_order]
       
        direction = mx.nd.zeros(7850)
        flip_vector = mx.nd.empty(epochs)
        # begin training        
        for e in range(epochs):
            # for each worker
            for i in range(num_workers):
                # sample a batch
                minibatch = np.random.choice(list(range(each_worker_data[i].shape[0])), size=batch_size, replace=False)
                # forward
                with autograd.record():
                    output = net(each_worker_data[i][minibatch])
                    loss = softmax_cross_entropy(output, each_worker_label[i][minibatch])
                # backward
                loss.backward()
                grad_list.append([param.grad().copy() for param in net.collect_params().values()])

            if args.aggregation == 'trim':
                # we aggregate the gradients instead of local model weights in this demo because for the 
                # aggregation rules in our setting, it is equivalent to aggregate either of them
                _, direction, flip_count = nd_aggregation.trim(e, grad_list, net, lr, byz, direction, args.nbyz, args.nbyz) 
                flip_vector[e] = flip_count
                
            elif args.aggregation == 'krum':
                _, direction, flip_count = nd_aggregation.krum(e, grad_list, net, lr, byz, direction, args.nbyz, args.nbyz) 
                flip_vector[e] = flip_count
                print (flip_count)
            else:
                sys.exit('Not Implemented aggregation!')
            
            # free memory
            del grad_list
            # reset the list
            grad_list = []
            
            # compute training accuracy every 10 iterations
            if (e+1) % 1 == 0:
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
                    
            # compute the final testing accuracy
            if (e+1) == args.nepochs:
                test_accuracy = evaluate_accuracy(test_data, net)
                print("Epoch %02d. Test_acc %0.4f" % (e, test_accuracy))
        np.save("Flip.npy", flip_vector)

if __name__ == "__main__":
    args = parse_args()
    main(args)
