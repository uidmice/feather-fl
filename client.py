import torch
from torch import nn
from torch.utils.data import DataLoader

from  model import MLP, QT

class Client:
    def __init__(self, dim_in, dim_hidden, dim_out, dataset, mode = 'FP'):
        self.mode = mode   # FP: FLOATING POINT, QAT: QUANTIZATION AWARE TRAINING
        self.dataset = dataset
 
        if mode == 'FP':
            self.model = MLP(dim_in, dim_hidden, dim_out)
            self.global_model = MLP(dim_in, dim_hidden, dim_out)
            
#         else:
#             self.model = QMLP(dim_in, dim_hidden, dim_out)
#             self.global_model = QMLP(dim_in, dim_hidden, dim_out)
        self.loss = nn.CrossEntropyLoss()
        
        
    def reset(self):
        self.model.reset()  
        self.global_model.load_state_dict(self.model.state_dict())
        
    def local_SGD(self, batch_size, steps, lr):
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        optim = torch.optim.SGD(self.model.parameters(), lr)

        tot_loss = 0 
        for j in range(round(steps * batch_size/len(self.dataset))):
            for i, data in enumerate(data_loader):
                inputs = data[0]
                labels = data[1]
                optim.zero_grad()
                output = self.model(inputs)
                loss = self.loss(output, labels)
        #             print(f"output: {output}, target: {y_batch}, loss: {loss.item()}")
                tot_loss += loss.item()
                loss.backward()
                optim.step()
        
        return tot_loss # no average across batch or steps
    
    def update_model(self, global_model):
        self.model.load_state_dict(global_model.state_dict())
        self.global_model.load_state_dict(global_model.state_dict())
        return
    
    def local_range_w(self):
        print(self.model.parameter_range())

    def global_range_w(self):
        print(self.global_model.parameter_range())
        
    def get_model_diff(self):
        state_dict = {}
        with torch.no_grad():
            for (na, param_a), (nb, param_b) in zip(self.model.named_parameters(), self.global_model.named_parameters()):
                if na == nb:
                    diff = param_a - param_b
                    state_dict[na] = diff
        return state_dict
