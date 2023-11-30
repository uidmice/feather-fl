#!/usr/bin/env python


import argparse, time, pickle, os
import numpy as np
from utils import  get_dataset, exp_setup
from algorithms import * 

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--total_ep', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--num_clients', type=int, default=5,
                        help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=60,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--participation', choices=[0,1,2, 3], type=int, default=0, help="0:cyclic, 1: deterministically spread, 2:random periodic, 3.random")
    parser.add_argument('--fl', choices=['feddif','fedavg','fedasync', 'fedaw','centralized', 'centralized_p'], default='fedaw', help='Fl algs: feddif, fedasync, fedavg, fedmis')
    parser.add_argument('--sync', action='store_true', help='synchronous FL')
    parser.add_argument('--model_sync', action='store_true', help='latest_model synced')

    # model arguments
    parser.add_argument('--training_mode', choices=['fp','qt'], default='qt', help='mode: fp (floating point), qt (quantized training)')
    parser.add_argument('--deterministic_rounding', action='store_false',dest='sr',  help='use deterministic rounding') 

    # other arguments
    parser.add_argument('--log', type=str, default='log', help='directry for tensorboard')
    parser.add_argument('--log_dir', type=str, default='', help='directry for saving data')
    parser.add_argument('--suffix', type=str, help='additional comments')

    parser.add_argument('--dataset', type=str, default='thermal', choices=['mnist','thermal', 'climate'], help="name \
                        of dataset: mnist, climate, thermal")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = args_parser()

    logger = None
    if not args.log_dir:
        args.log_dir = args.dataset
    dir_path = f"{args.log_dir}/{args.fl}_{args.training_mode}_part_{args.participation}_lr_{args.lr}"
    if args.suffix:
        dir_path += f"_{args.suffix}"
    if not os.path.exists('runs'):
        os.makedirs('runs')
    if not os.path.exists(f'runs/{args.log_dir}'):
        os.makedirs(f'runs/{args.log_dir}')    
    if not os.path.exists(f'runs/{dir_path}'):
        os.makedirs(f'runs/{dir_path}')    
    if args.log:
        from torch.utils.tensorboard import SummaryWriter
        logger = SummaryWriter(f'{args.log}/{dir_path}')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cpu'

    # if torch.cuda.is_available() :
    #     torch.cuda.manual_seed(args.seed)
    #     device = 'cuda' 


    test_dataset, client_train_dataset = get_dataset(args)

    global_model, clients = exp_setup(args, client_train_dataset)

    start_time = time.time()
    if args.fl == 'centralized': #for sanity check
        train_loss, test_loss, test_accuracy = train_global(global_model, clients,test_dataset, args, logger)

    elif args.fl == 'centralized_p':
        train_loss, test_loss, test_accuracy = train_global_partitioned(global_model, clients,test_dataset, args, logger)

    elif args.fl == 'fedavg':
        train_loss, test_loss, test_accuracy = train_FedAvg(global_model, clients,test_dataset, args, logger)

    elif args.fl == 'feddif' or args.fl == 'fedasync' :
        train_loss, test_loss, test_accuracy = train_FedAsync(global_model, clients,test_dataset, args, logger)
    elif args.fl == 'fedaw':
        train_loss, test_loss, test_accuracy, global_model = train_FedAW(global_model, clients,test_dataset, args, logger)
    else:
        raise ValueError(f"{args.fl} not implemented")

    if logger:
        logger.close()

    print(f' \n Results after {args.total_ep} global rounds of training:')
    print("|---- Avg Train Loss: {:.2f}".format(np.average(train_loss[-args.num_clients])))
    print("|---- Test Loss: {:.2f}".format(test_loss[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_accuracy[-1]))
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    print(global_model.parameter_range())


    pickle.dump([test_loss, test_accuracy, train_loss], open(f'runs/{dir_path}/results.pk', 'wb'))
    pickle.dump(global_model, open(f'runs/{dir_path}/model.pk', 'wb'))
    pickle.dump(args, open(f'runs/{dir_path}/args.pk', 'wb'))


