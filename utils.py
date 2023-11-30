import torch
import pickle
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import  numpy as np
from model import MLP, QT
from client import Client

def get_dataset(args):
    if args.num_clients <6:
        if args.dataset == 'mnist':
            test_ds = pickle.load(open('data/mnist/test_data.pk', 'rb'))
            client_ds = pickle.load(open('data/mnist/client_data.pk', 'rb'))
        elif args.dataset == 'thermal':
            test_ds = pickle.load(open('data/thermal/test_data.pk', 'rb'))
            client_ds = pickle.load(open('data/thermal/client_data.pk', 'rb'))
            if args.training_mode == 'fp':
                td = []
                for data in test_ds:
                    td.append((data[0] * 255/70, data[1]))
                test_ds = td
                cd = []
                for client in client_ds:
                    td = []
                    for data in client:
                        td.append((data[0] * 255/70, data[1]))
                    cd.append(td)
                client_ds = cd
        else:
            raise ValueError(f"dataset {args.dataset} not provided")

    else:
        if args.dataset == 'mnist':
            test_ds, _, client_ds, _ = prepare_mnist_data(dim=7, iid=0, 
                                                            num_dev = args.num_clients, 
                                                            num_ds = 300)
        else:
            raise ValueError(f"Only MNIST for a large number of clients")

    return test_ds, client_ds


def exp_setup(args, client_data):
    if 'centralized' in args.fl:
        args.training_mode = 'fp'
        
    if args.dataset == 'mnist':
        input_dim = 49
        hidden_dim = 32
        output_dim = 2
        clients = [Client(input_dim, hidden_dim, output_dim, cdata, args.training_mode) for cdata in client_data]


    elif args.dataset == 'thermal':
        input_dim = 64
        hidden_dim = 32
        output_dim = 2
        args.local_bs = 80

        clients = [Client(input_dim, hidden_dim, output_dim, cdata, 
                          args.training_mode, 0.5, 1.2, 5.0) for cdata in client_data]

    else:
        raise ValueError(f"dataset {args.dataset} not provided")

    global_model = MLP(input_dim, hidden_dim, output_dim)
    # clients = [Client(input_dim, hidden_dim, output_dim, cdata, args.training_mode) for cdata in client_data]

    return global_model, clients

def find_data_iter(steps, batch_size, ds_size):
    return int((steps * batch_size - 1) / ds_size + 1) 

def test_inference(model, dataset, loss):
    model.eval()
    l, correct = 0.0, 0.0

    testloader = DataLoader(dataset, batch_size=2000, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(testloader):
            x = data[0]
            labels = data[1]
            # Inference
            outputs = model(x)
            batch_loss = loss(outputs, labels)
            l += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()

    return correct/len(dataset), l


def mnist_iid_samples(dataset, num_users, subset_size):
    total_indices = np.arange(len(dataset))
    np.random.shuffle(total_indices)

    ds = []
    sub_idx =[]
    for i in range(num_users):
        indices = total_indices[i*subset_size:(i+1)*subset_size]
        sub_idx.append(indices)
        original = [dataset[i] for i in indices]
        ds.append([(x,label % 2, label) for (x,label) in original])
    return ds, sub_idx


def mnist_non_iid_samples(dataset, num_users, sub_size):
    total_indices = np.arange(len(dataset))
    np.random.shuffle(total_indices)

    ds = []
    sub_idx =[]
    for i in range(num_users):
        subset_label = np.random.choice(range(10), 6, replace=False)
        subset_idx =  [i for i, (x,y) in enumerate(dataset) if y in subset_label]
        indices = np.random.choice(subset_idx, sub_size, replace=False)
        sub_idx.append(indices)
        original = [dataset[i] for i in indices]
        ds.append([(x,label % 2, label) for (x,label) in original])
    return ds, sub_idx

    
    
def prepare_mnist_data(dim, iid=0, num_dev = 5, num_ds = 300):

    transform = transforms.Compose([
        transforms.Resize((dim, dim)),  # Resize the image to 14x14
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    mnist_train = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('data/mnist', train=False, download=True, transform=transform)

    mnist_even_odd_test= [(x,label % 2, label) for (x,label) in mnist_test]
    mnist_even_odd_train= [(x,label % 2, label) for (x,label) in mnist_train]

    if iid:
        return mnist_even_odd_test, mnist_even_odd_train, *mnist_iid_samples(mnist_train, num_dev, num_ds)
    return mnist_even_odd_test, mnist_even_odd_train, *mnist_non_iid_samples(mnist_train, num_dev, num_ds)

def samples(dataset, num_users):
    total_indices = np.arange(len(dataset))
    np.random.shuffle(total_indices)

    ds = []
    num_data = len(dataset) // num_users
    for i in range(num_users):
        indices = total_indices[i*num_data:(i+1)*num_data]
        original = [dataset[i] for i in indices]
        ds.append([(x,label) for (x,label) in original])
    return ds

def prepare_thermal_data(num_dev = 5, num_ds = 300):

    train_dataset = torch.load('data/thermal/train_dataset_scaled.pt')
    test_dataset = torch.load('data/thermal/test_dataset_scaled.pt')

    return test_dataset, train_dataset, samples(train_dataset, num_dev)
