import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import numpy as np
import sys
import pdb
from copy import deepcopy
import aggregation
import attack

# abcd

## Read the command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset", default='mnist', type=str)
    parser.add_argument("--bias", help="degree of non-IID to assign data to workers", type=float, default=0.5)
    parser.add_argument("--net", help="net", default='dnn', type=str, choices=['mlr', 'dnn', 'resnet18'])
    parser.add_argument("--batch_size", help="batch size", default=32, type=int)
    parser.add_argument("--lr", help="learning rate", default=0.01, type=float)
    parser.add_argument("--nworkers", help="# workers", default=10, type=int)
    parser.add_argument("--nepochs", help="# epochs", default=50, type=int)
    parser.add_argument("--gpu", help="index of gpu", default=-1, type=int)
    parser.add_argument("--seed", help="seed", default=42, type=int)
    parser.add_argument("--nbyz", help="# byzantines", default=2, type=int)
    parser.add_argument("--byz_type", help="type of attack", default='benign', type=str,
                        choices=['benign', 'full_trim', 'full_krum', 'adaptive_trim', 'adaptive_krum', 'shej_attack', 'shej_agnostic'])
    parser.add_argument("--aggregation", help="aggregation rule", default='fedsgd', type=str)
    parser.add_argument("--cmax", help="FLAIR's notion of c_max", default=2, type=int)
    parser.add_argument("--decay", help="Decay rate", default=0.99, type=float)
    parser.add_argument("--exp", help="Experiment name", default='', type=str)
    return parser.parse_args()

## Defining class for ResNet-18 implementation
class PreActBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x

        return out + shortcut


class ResNet18(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            self._make_layer(64, 64, num_blocks[0], stride=1),
            self._make_layer(64, 128, num_blocks[1], stride=2),
            self._make_layer(128, 256, num_blocks[2], stride=2),
            self._make_layer(256, 256, num_blocks[3], stride=2),
        )

        self.classifier = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(PreActBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)

        x = self.layers(x)

        x_avg = F.adaptive_avg_pool2d(x, (1, 1))
        x_avg = x_avg.view(x_avg.size(0), -1)

        x_max = F.adaptive_max_pool2d(x, (1, 1))
        x_max = x_max.view(x_max.size(0), -1)

        x = torch.cat([x_avg, x_max], dim=-1)

        x = self.classifier(x)

        return x

#Multi-class logistic regression
class MLR(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(MLR, self).__init__()
        self.linear = nn.Linear(inp_dim, out_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out

#DNN
class DNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 30, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(30, 50, 3, padding=1)
        self.fc1 = nn.Linear(50*7*7, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 50*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#Learning rate scheduler used to train CIFAR-10
def get_lr(epoch, num_epochs, lr):

    mu = num_epochs/4
    sigma = num_epochs/4
    max_lr = lr
    if (epoch < num_epochs/4):
        return max_lr*(1-np.exp(-25*(epoch/num_epochs)))
    else:
        return max_lr*np.exp(-0.5*(((epoch-mu)/sigma)**2))

           
def main(args):
    
    num_workers = args.nworkers
    num_epochs = args.nepochs
    
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    batch_size = args.batch_size
    lr = args.lr
    filename = args.exp

    ###Load datasets
    if (args.dataset == 'mnist'):
        transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]) 
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download='True', transform=transform)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download='True', transform=transform)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        del trainset, testset        
        num_inputs = 28 * 28
        num_outputs = 10

    elif args.dataset == 'cifar10':
        num_inputs = 32*32*3
        num_outputs = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download='True', transform=transform_train)
        train_data = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download='True', transform=transform_test)
        test_data = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        del trainset, testset
        
    else:
        sys.exit('Not Implemented Dataset!')
        
    ####Load models
    if (args.net == 'mlr'):
        net = MLR(num_inputs, num_outputs)
    elif (args.net == 'resnet18'):
        net = ResNet18()
    elif(args.net == 'dnn'):
        net = DNN()
        
    net.to(device) # -------
    
    if args.byz_type == 'benign':
        byz = attack.benign
    elif args.byz_type == 'full_trim':
        byz = attack.full_trim
    elif args.byz_type == 'full_krum':
        byz = attack.full_krum
    elif args.byz_type == 'adaptive_trim':
        byz = attack.adaptive_trim
    elif args.byz_type == 'adaptive_krum':
        byz = attack.adaptive_krum
    elif args.byz_type == 'shej_attack':
        byz = attack.shej_attack
    elif args.byz_type == 'shej_agnostic':
        byz = attack.shej_agnostic

    ####Distribute data samples according to a given non-IID bias
    if args.aggregation == 'fltrust': num_workers=num_workers+1 #one extra worker containing the root dataset
    bias_weight = args.bias
    other_group_size = (1-bias_weight) / (num_outputs-1)
    worker_per_group = num_workers / (num_outputs)
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)] 
    for _, (data, label) in enumerate(train_data):
        if args.net == 'mlr':
            data = data.reshape((-1, num_inputs))
        for (x, y) in zip(data, label):
            upper_bound = (y.item()) * (1-bias_weight) / (num_outputs-1) + bias_weight
            lower_bound = (y.item()) * (1-bias_weight) / (num_outputs-1)
            rd = np.random.random_sample()
            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size)+y.item()+1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.item()

            # assign a data point to a worker
            rd = np.random.random_sample()
            selected_worker = int(worker_group*worker_per_group + int(np.floor(rd*worker_per_group)))
            if (args.bias == 0): selected_worker = np.random.randint(num_workers)
            each_worker_data[selected_worker].append(x.to(device))
            each_worker_label[selected_worker].append(y.to(device))

    # concatenate the data for each worker
    each_worker_data = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_data] 
    each_worker_label = [(torch.stack(each_worker, dim=0)).squeeze(0) for each_worker in each_worker_label]
    
    # random shuffle the workers
    random_order = np.random.RandomState(seed=42).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    #define weights proportional to data size of a client for FEDSGD
    wts = torch.zeros(len(each_worker_data)).to(device)
    for i in range(len(each_worker_data)):
        wts[i] = len(each_worker_data[i])
    wts = wts/torch.sum(wts)
    criterion = nn.CrossEntropyLoss()
    test_acc = np.empty(num_epochs)
    
    #Count the number of parameters
    P = 0
    for param in net.parameters():
        if param.requires_grad:
            P = P + param.nelement()

    direction = torch.zeros(P).to(device)
    susp = torch.zeros(num_workers).to(device)
    decay = args.decay
    
    batch_idx = np.zeros(num_workers)
    susp_score = []
    new_flips = []
    faba_client_list = []
    fg_client_list = []
    flt_client_list = []
    weight = torch.ones(num_workers)
    for epoch in range(num_epochs):
        grad_list = []
        if (args.aggregation == 'flair'):
            susp = susp*decay #suspicion score only used by FLAIR
        if (args.dataset == 'cifar10'):
            lr = get_lr(epoch, num_epochs, args.lr)
        for worker in range(num_workers):
            net_local = deepcopy(net) # --------------------------------------------------------------------------------------------------------------------------------------
            net_local.train()
            optimizer = optim.SGD(net_local.parameters(), lr=lr)
            optimizer.zero_grad()
            
            #sample local dataset in a round-robin manner
            if (batch_idx[worker]+batch_size < each_worker_data[worker].shape[0]):
                minibatch = np.asarray(list(range(int(batch_idx[worker]),int(batch_idx[worker])+batch_size)))
                batch_idx[worker] = batch_idx[worker] + batch_size
            else: 
                minibatch = np.asarray(list(range(int(batch_idx[worker]),each_worker_data[worker].shape[0]))) 
                batch_idx[worker] = 0
            output = net_local(each_worker_data[worker][minibatch].to(device))
            loss = criterion(output, each_worker_label[worker][minibatch].to(device))
            loss.backward()
            optimizer.step()
                    
            ##append all gradients in a list
            grad_list.append([(x-y).detach() for x, y in zip(net_local.parameters(), net.parameters()) if x.requires_grad != 'null'])
            
            del net_local, output, loss
            torch.cuda.empty_cache()
        
        ###Do the aggregation
        if (args.aggregation == 'fedsgd'):
            net = aggregation.FEDSGD(device, byz, lr, grad_list, net, args.nbyz, wts) 

        elif (args.aggregation == 'flair'):
            if (epoch == 0): flip_new = torch.ones(num_workers) ##initializing flip-score cutoff to 1
            else:
                fs_cut = torch.sort(flip_new)[0][args.nworkers-args.nbyz-1]
            net, direction, susp, flip_new, weight = aggregation.flair(device, byz, lr, grad_list, net, direction, susp, flip_new, args.cmax, weight)
            if byz=='benign': actual_c = 0
            else: actual_c = args.nbyz
            new_flips.append(flip_new.cpu().numpy())
            susp_score.append(susp.cpu().numpy())
        elif (args.aggregation == 'krum'):
            net = aggregation.krum(device, byz, lr, grad_list, net, args.nbyz)         
        elif (args.aggregation == 'trim'):
            net = aggregation.trim(device, byz, lr, grad_list, net, args.cmax)
        elif (args.aggregation == 'faba'):
            net, faba_list = aggregation.faba(device, byz, lr, grad_list, net, args.cmax)    
            faba_client_list.append(faba_list)
        elif (args.aggregation == 'foolsgold'):
            net, fg_list = aggregation.foolsgold(device, byz, lr, grad_list, net, args.cmax)
            fg_client_list.append(fg_list.cpu().numpy())
        elif (args.aggregation == 'fltrust'):
            net, flt_list = aggregation.fltrust(device, byz, lr, grad_list, net, args.nbyz)
            flt_client_list.append(flt_list.cpu().numpy())
        elif (args.aggregation == 'median'):
            net = aggregation.median(device, byz, lr, grad_list, net, args.nbyz)

        del grad_list
        torch.cuda.empty_cache()
        
        ##Evaluate the learned model on test dataset
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_data:
                images, labels = data
                if (args.net == 'mlr'):
                    images = images.reshape((-1, num_inputs))
                outputs = net(images.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()
            test_acc[epoch] = correct/total                
            print ('Iteration: %d, test_acc: %f' %(epoch, test_acc[epoch]))      
    np.save(filename+'_test_acc.npy', test_acc)
    torch.save(net.state_dict(), filename+'_model.pth')
    if (args.aggregation == 'fltrust'):
        np.save(filename+'_FL_list.npy', np.asarray(flt_client_list))
    if (args.aggregation == 'flair'):
        np.save(filename+'_newFS.npy', np.asarray(new_flips))
        np.save(filename+'_susp.npy', np.asarray(susp_score))        
    if (args.aggregation == 'faba'):
        np.save(filename+'_faba_list.npy', np.asarray(faba_client_list))
    if (args.aggregation == 'foolsgold'):
        np.save(filename+'_FG_list.npy', np.asarray(fg_client_list))
if __name__ == "__main__":
    args = parse_args()
    main(args)
