import torch
import random
from torch import nn
from utils import test_inference, find_data_iter
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
from model import MLP, QT, stochastic_round

def update_global_model_FedDiff(gm, client, num_dev):
    state_dict_c = client.get_model_diff()
    with torch.no_grad():
        if client.mode == 'fp':
            for name, param in gm.named_parameters():
                param.add_(state_dict_c[name]/num_dev) 
        else:
            state_dict = gm.state_dict()
            for name in state_dict:
                state_dict[name] += state_dict_c[name].t() /num_dev * client.model.scale_dict[name] 
            gm.load_state_dict(state_dict)

            
def update_global_model_FedAvg(gm, clients):
    with torch.no_grad():
        for param in gm.parameters():
            param.zero_()
        state_dict = gm.state_dict()

        for client in clients:
            state_dict_c = client.model.state_dict()
            for name, param in gm.named_parameters():
                if client.mode == 'fp':
                    state_dict[name] += state_dict_c[name]/len(clients)
                else:
                    state_dict[name] += state_dict_c[name].t() * client.model.scale_dict[name]/len(clients)
        gm.load_state_dict(state_dict)
    

def update_global_model_FedAsync(gm, client, dt):
    alpha = 1/(1 + dt)
    state_dict = gm.state_dict()
    state_dict_c = client.model.state_dict()

    with torch.no_grad():
        for name in state_dict:
            if client.mode == 'fp':
                state_dict[name] = (1 - alpha) * state_dict[name] + alpha * state_dict_c[name]
            else:
                state_dict[name] = (1 - alpha) * state_dict[name] + alpha * state_dict_c[name].t() * client.model.scale_dict[name]
        gm.load_state_dict(state_dict)


def update_global_model_FedAW(cms, client, dts, i):
    gm = MLP(client.model.dim_in, client.model.dim_hidden, client.model.dim_out)

    alphas = [1/(1 + dt)**2 for dt in dts]
    w = np.sum(alphas)
    with torch.no_grad():
        for param in gm.parameters():
            param.zero_()
        state_dict = gm.state_dict()
        for cm, alpha in zip(cms, alphas):
            for name in state_dict:
                if client.mode == 'fp':
                    state_dict[name] += alpha * cm[name] / w
                else:
                    state_dict[name] += alpha * cm[name].t() * client.model.scale_dict[name] / w
        state_dict_c = client.get_model_diff()
        # alpha = alphas[i]
        alpha = 1/len(cms)
        for name in state_dict:
            if client.mode == 'fp':
                state_dict[name] += alpha * state_dict_c[name]
            else:
                state_dict[name] += alpha * state_dict_c[name].t() * client.model.scale_dict[name]
 
    gm.load_state_dict(state_dict)
    return gm

def train_FedAW(model, clients, test_data, args, logger=None):
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

    last_update_time = [-1 for i in range(len(clients))]


    for t in range(args.total_ep):
        if args.participation == 1:
            order = permu[t%len(clients)]
        elif args.participation == 2:
            random.shuffle(order)
        
        if args.sync:
            for client in clients: 
                client.update_model(model)

        for i, j in enumerate(order):
            if args.participation == 3:
                j = random.randint(0, len(clients)-1)
            client = clients[j]
            if args.model_sync:
                client.update_model(model)
            
            cur_time = t * args.num_clients + i
            if args.training_mode == 'fp':
                batch_loss = client.local_SGD(args.local_bs , args.local_ep, args.lr)
            else:
                batch_loss = client.local_SGD_GT(args.local_bs , args.local_ep, args.lr, args.sr)
            model = update_global_model_FedAW([client.global_model.state_dict() for client in clients], client, 
                                              [cur_time - t - 1 for t in last_update_time], j)
            
            last_update_time[j] = cur_time
            client.update_model(model)
            train_loss.append(batch_loss)
            acc, l = test_inference(model, test_data, CEloss)
            test_accuracy.append(acc)
            test_loss.append(l)
            print("|---- Train Loss: {:.2f}".format(batch_loss))

            if logger:
                logger.add_scalar("Train/Loss", batch_loss, t * args.num_clients + i)
                logger.add_scalar("Test/Acc", acc, t * args.num_clients + i)
                logger.add_scalar("Test/Loss", l, t * args.num_clients + i)
        print(f"|---- Global Step: {t}")
        print("|---- Test Loss: {:.2f}".format(l))
        print("|---- Test Accuracy: {:.2f}%".format(100*acc))
        print(f"|----------------------------------------------|")

    return  train_loss, test_loss, test_accuracy, model

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

    last_update_time = [-1 for i in range(len(clients))]


    for t in range(args.total_ep):
        if args.participation == 1:
            order = permu[t%len(clients)]
        elif args.participation == 2:
            random.shuffle(order)
        
        if args.sync:
            for client in clients: 
                client.update_model(model)

        for i, j in enumerate(order):
            if args.participation == 3:
                j = random.randint(0, len(clients)-1)
            client = clients[j]
            if args.model_sync:
                client.update_model(model)
            
            cur_time = t * args.num_clients + i
            if args.training_mode == 'fp':
                batch_loss = client.local_SGD(args.local_bs , args.local_ep, args.lr)
            else:
                batch_loss = client.local_SGD_GT(args.local_bs , args.local_ep, args.lr, args.sr)
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
            print("|---- Train Loss: {:.2f}".format(batch_loss))

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
                batch_loss = client.local_SGD_GT(args.local_bs , args.local_ep, args.lr, args.sr)            
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
    if args.participation == 1:
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
            if args.participation == 3:
                j = random.randint(0, len(clients)-1)
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