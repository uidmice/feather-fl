import torch
import random
from torch import nn
from utils import test_inference, find_data_iter
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

def update_global_model_FedDiff(gm, client, num_dev):
    state_dict_c = client.get_model_diff()
    with torch.no_grad():
        if client.mode == 'fp':
            for name, param in gm.named_parameters():
                param.add_(state_dict_c[name]/num_dev) 
        else:
            state_dict = gm.state_dict()
            state_dict['l1.weight'] += state_dict_c['w1'].t() * client.model.scale_dict['w1']/num_dev
            state_dict['l2.weight'] += state_dict_c['w2'].t() * client.model.scale_dict['w2']/num_dev
            state_dict['l1.bias'] += state_dict_c['b1'].t() * client.model.scale_dict['b1']/num_dev
            state_dict['l2.bias'] += state_dict_c['b2'].t() * client.model.scale_dict['b2']/num_dev
            gm.load_state_dict(state_dict)

            
def update_global_model_FedAvg(gm, clients):
    with torch.no_grad():
        for param in gm.parameters():
            param.zero_()
        state_dict = gm.state_dict()

        for client in clients:
            state_dict_c = client.model.state_dict()
            if client.mode == 'fp':
                for name, param in gm.named_parameters():
                    state_dict[name] += state_dict_c[name]/len(clients)
            else:
                state_dict['l1.weight'] += state_dict_c['w1'].t() * client.model.scale_dict['w1']/len(clients)
                state_dict['l2.weight'] += state_dict_c['w2'].t() * client.model.scale_dict['w2']/len(clients)
                state_dict['l1.bias'] += state_dict_c['b1'].t() * client.model.scale_dict['b1']/len(clients)
                state_dict['l2.bias'] += state_dict_c['b2'].t() * client.model.scale_dict['b2']/len(clients)
        gm.load_state_dict(state_dict)
    

def update_global_model_FedAsync(gm, client, dt):
    alpha = 1/(1 + dt)
    state_dict = gm.state_dict()
    state_dict_c = client.model.state_dict()

    with torch.no_grad():
        if client.mode == 'fp':
            for name in state_dict:
                state_dict[name] *= 1 - alpha
                state_dict[name] += alpha * state_dict_c[name]
        else:
            state_dict['l1.weight'] = state_dict['l1.weight'] * (1-alpha) + alpha * state_dict_c['w1'].t() * client.model.scale_dict['w1']
            state_dict['l2.weight'] = state_dict['l2.weight'] * (1-alpha) + alpha *state_dict_c['w2'].t() * client.model.scale_dict['w2']
            state_dict['l1.bias'] = state_dict['l1.bias'] * (1-alpha) + alpha *state_dict_c['b1'].t() * client.model.scale_dict['b1']
            state_dict['l2.bias'] = state_dict['l2.bias'] * (1-alpha) + alpha *state_dict_c['b2'].t() * client.model.scale_dict['b2']
        gm.load_state_dict(state_dict)

        

def train_FedAsync(model, clients, test_data, args, logger=None):
    model.reset()
    for client in clients:
        client.update_model(model)
    
    test_accuracy = []
    test_loss = []
    train_loss = []
    CEloss = nn.CrossEntropyLoss()
    order = [i for i in range(args.num_clients)]
    
    if args.participation == 1:
        permu = []
        cur = [i for i in range(args.num_clients)]
        for i in range(len(clients)):
            permu.append(cur.copy())
            cur = [cur[-1]] + cur[:-1]
        random.shuffle(permu)

    last_update_time = [0 for i in range(len(clients))]


    for t in range(args.total_ep):
        if args.participation == 1:
            order = permu[t%len(clients)]
        elif args.participation == 2:
            random.shuffle(order)
        
        if args.sync:
            for client in clients: 
                client.update_model(model)

        for i, j in enumerate(order):
            client = clients[j]
            if args.model_sync:
                client.update_model(model)
            
            cur_time = t * args.num_clients + i
            if args.training_mode == 'fp':
                batch_loss = client.local_SGD(args.local_bs , args.local_ep, args.lr)
            else:
                batch_loss = client.local_SGD_GT(args.local_bs , args.local_ep, args.lr)
            if args.fl == 'feddif':
                update_global_model_FedDiff(model, client, len(clients))
            elif args.fl == 'fedasync':
                update_global_model_FedAsync(model, client, cur_time - last_update_time[j] - 1)
            
            last_update_time[j] = cur_time
            client.update_model(model)
            train_loss.append(batch_loss)
            acc, l = test_inference(model, test_data, CEloss)
            test_accuracy.append(acc)
            test_loss.append(l)
            if logger:
                logger.add_scalar("Train/Loss", batch_loss, t * args.num_clients + i)
                logger.add_scalar("Test/Acc", acc, t * args.num_clients + i)
                logger.add_scalar("Test/Loss", l, t * args.num_clients + i)
        print(f"|---- Global Step: {t}")
        print("|---- Test Loss: {:.2f}".format(l))
        print("|---- Test Accuracy: {:.2f}%".format(100*acc))
        print(f"|----------------------------------------------|")

    return  train_loss, test_loss, test_accuracy
            


def train_FedAvg(model, clients, test_data, args, logger=None):
    model.reset()
    for client in clients:
        client.reset()
    
    test_accuracy = []
    test_loss = []
    train_loss = []
    CEloss = nn.CrossEntropyLoss()


    for i in range (args.total_ep):
        for client in clients: 
            client.update_model(model)
        for client in clients:
            if args.training_mode == 'fp':
                batch_loss = client.local_SGD(args.local_bs , args.local_ep, args.lr)
            else:
                batch_loss = client.local_SGD_GT(args.local_bs , args.local_ep, args.lr)            
            train_loss.append(batch_loss)
        update_global_model_FedAvg(model, clients)
        acc, l = test_inference(model, test_data, CEloss)
        test_accuracy.append(acc)
        test_loss.append(l)
        print(f"|---- Global Step: {i}")
        print("|---- Test Loss: {:.2f}".format(l))
        print("|---- Test Accuracy: {:.2f}%".format(100*acc))
        print(f"|----------------------------------------------|")

        if logger:
            logger.add_scalar("Train/Loss", np.average(train_loss[-args.num_clients:]), (i +1) * args.num_clients )
            logger.add_scalar("Test/Acc", acc, (i +1) * args.num_clients )
            logger.add_scalar("Test/Loss", l, (i +1) * args.num_clients )
    return  train_loss, test_loss, test_accuracy

def train_global(model, clients, test_data, args, logger=None):
    model.reset()

    dataset_train = ConcatDataset([client.dataset for client in clients])
    data_loader = DataLoader(dataset_train, batch_size=args.local_bs, shuffle=True)
    optim = torch.optim.SGD(model.parameters(), args.lr)
    CEloss = nn.CrossEntropyLoss()
    

    test_accuracy = []
    test_loss = []
    train_loss = []
    
    step_count = 0

    loss_sum = 0        
    
    for t in range(find_data_iter(args.total_ep * args.local_ep * args.num_clients, args.local_bs, len(dataset_train))):
        for i, data in enumerate(data_loader):
            inputs = data[0]
            labels = data[1]
            optim.zero_grad()

            output = model(inputs)
            loss = CEloss(output, labels)
            loss.backward()
            loss_sum += loss.item()
            optim.step()
            
            step_count += 1

            if step_count % args.local_ep == 0:
                acc, l = test_inference(model, test_data, CEloss)
                test_accuracy.append(acc)
                test_loss.append(l)
                train_loss.append(loss_sum)
                loss_sum = 0

                if logger:
                    logger.add_scalar("Train/Loss", loss_sum/args.local_ep, step_count//args.local_ep)
                    logger.add_scalar("Test/Acc", acc, step_count//args.local_ep)
                    logger.add_scalar("Test/Loss", l, step_count//args.local_ep)
                if step_count//args.local_ep % args.num_clients == 0:
                    print(f"|---- Global Step: {step_count//args.local_ep//args.num_clients}")
                    print("|---- Test Loss: {:.2f}".format(l))
                    print("|---- Test Accuracy: {:.2f}%".format(100*acc))   
                    print("|----------------------------------------------|")
 
    return  train_loss, test_loss, test_accuracy




def train_global_partitioned(model, clients, test_data, args, logger=None):
    model.reset()
    client_data = [client.dataset for client in clients]
    data_loaders = [DataLoader(dataset_train, batch_size=args.local_bs, shuffle=True) for dataset_train in client_data]
    optim = torch.optim.SGD(model.parameters(), args.lr)
    CEloss = nn.CrossEntropyLoss()
    

    test_accuracy = []
    test_loss = []
    train_loss = []
    
    order = [i for i in range(args.num_clients)]
    permu = []
    cur = [i for i in range(args.num_clients)]
    for i in range(len(clients)):
        permu.append(cur.copy())
        cur = [cur[-1]] + cur[:-1]
    random.shuffle(permu)

    for t in range(args.total_ep):
        if args.participation == 1:
            order = permu[t%len(clients)]
        elif args.participation == 2:
            random.shuffle(order)

        for i, j in enumerate(order):
            data_loader = data_loaders[j]
            tot_loss = 0
            for j in range(find_data_iter(args.local_ep, args.local_bs, len(client_data[0]))):
                for _, data in enumerate(data_loader):
                    inputs = data[0]
                    labels = data[1]
                    optim.zero_grad()

                    output = model(inputs)
                    loss = CEloss(output, labels)
                    tot_loss += loss.item()
                    loss.backward()
                    optim.step()
            # tot_loss/=args.local_ep
            train_loss.append(tot_loss)
            acc, l = test_inference(model, test_data, CEloss)
            test_accuracy.append(acc)
            test_loss.append(l)
            if logger:
                logger.add_scalar("Train/Loss", tot_loss, t * args.num_clients + i)
                logger.add_scalar("Test/Acc", acc, t * args.num_clients + i)
                logger.add_scalar("Test/Loss", l, t * args.num_clients + i)
        print(f"|---- Global Step: {t }")
        print("|---- Test Loss: {:.2f}".format(l))
        print("|---- Test Accuracy: {:.2f}%".format(100*acc))
        print("|----------------------------------------------|")


    return  train_loss, test_loss, test_accuracy