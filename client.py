import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from  model import MLP, QT, stochastic_round



class Client:
    def __init__(self, dim_in, dim_hidden, dim_out, dataset, mode = 'fp', w1_range = None, w2_range = None, y_range = None):
        self.mode = mode   # FP: FLOATING POINT, qt: QUANTIZATION AWARE TRAINING
        self.dataset = dataset
 
        if mode == 'fp':
            self.model = MLP(dim_in, dim_hidden, dim_out)
            self.global_model = MLP(dim_in, dim_hidden, dim_out)


            
        else: #qt
            self.model = QT(dim_in, dim_hidden, dim_out, w1_range, w2_range, y_range)
            self.global_model = QT(dim_in, dim_hidden, dim_out, w1_range, w2_range, y_range)
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
    
    def local_SGD_GT(self, batch_size, steps, lr, sr = True):
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        tot_loss = 0 
        for j in range(round(steps * batch_size/len(self.dataset))):
            for i, data in enumerate(data_loader):
                inputs = data[0]
                labels = data[1]
                if sr:
                    inputs = stochastic_round((inputs * 255))
                else:
                    inputs = (inputs * 255).to(torch.int)
                # bs x hidden_dim
                y = F.relu(torch.matmul(inputs , self.model.w1) + self.model.b1)
                y = torch.clamp(y * self.model.s_b1/self.model.s_y, min=0, max=255)
                if sr:
                    y = stochastic_round(y)
                else:
                    y = torch.round(y).to(torch.int)
                # bs x output_dim
                output = (torch.matmul(y , self.model.w2) + self.model.b2) * self.model.s_b2
                with torch.no_grad():
                    loss = self.loss(output, labels)
                    tot_loss += loss.item()
                #bs x output_dim
                
                output = F.softmax(output, dim=-1)
                if sr:
                    dl = stochastic_round((output - F.one_hot(labels, 2)) / self.model.b2_grad_s)
                else:
                    dl = torch.round((output - F.one_hot(labels, 2)) / self.model.b2_grad_s).to(torch.int)

                # (output, )
                b2_grad = torch.sum(dl, dim=0)

                # (hidden_dim  x output_dim)
                w2_grad = torch.matmul(y.t(), dl)

                # bs x hidden_dim
                dy = torch.matmul(dl, self.model.w2.t() )
                dy = torch.where(y > 0, dy, torch.tensor(0))

                # (hidden_dim, )
                b1_grad = torch.sum(dy, dim=0)
                # (input_dim  x hidden_dim)
                w1_grad = torch.matmul(inputs.t(), dy)

                w1 = self.model.w1.to(torch.float) - w1_grad * self.model.w1_grad_s * lr / len(inputs) / self.model.s_w1
                w2 = self.model.w2.to(torch.float) - w2_grad * self.model.w2_grad_s * lr / len(inputs) / self.model.s_w2
                b1 = self.model.b1.to(torch.float) - b1_grad * self.model.b1_grad_s * lr / len(inputs) / self.model.s_b1
                b2 = self.model.b2.to(torch.float) - b2_grad * self.model.b2_grad_s * lr / len(inputs) / self.model.s_b2

                # stochastic rounding
                if sr:
                    self.model.w1 = torch.clamp(stochastic_round(w1), -127, 127).to(torch.int)
                    self.model.w2 = torch.clamp(stochastic_round(w2), -127, 127).to(torch.int)
                    self.model.b1 = torch.clamp(stochastic_round(b1), -20/self.model.s_b1, 20/self.model.s_b1).to(torch.int)
                    self.model.b2 = torch.clamp(stochastic_round(b2), -20/self.model.s_b1, 20/self.model.s_b2).to(torch.int)
                else:
                    self.model.w1 = torch.round(w1).to(torch.int)
                    self.model.w2 = torch.round(w2).to(torch.int)
                    self.model.b1 = torch.round(b1).to(torch.int)
                    self.model.b2 = torch.round(b2).to(torch.int)   
        return tot_loss


    def update_model(self, global_model):
        if self.mode == 'qt':
            gmq = global_model.quantize()
            gm_state = gmq.state_dict()
        else:
            gm_state = global_model.state_dict()
        self.global_model.load_state_dict(gm_state)
        self.model.load_state_dict(gm_state)
        return
    
    def local_range_w(self):
        print(self.model.parameter_range())

    def global_range_w(self):
        print(self.global_model.parameter_range())
        
    def get_model_diff(self):
        state_dict = {}
        if self.mode == 'fp':
            with torch.no_grad():
                for (na, param_a), (nb, param_b) in zip(self.model.named_parameters(), self.global_model.named_parameters()):
                    if na == nb:
                        diff = param_a - param_b
                        state_dict[na] = diff
        else:
            for (na, param_a), (nb, param_b) in zip(self.model.state_dict().items(), self.global_model.state_dict().items()):
                if na == nb:
                    diff = param_a - param_b
                    state_dict[na] = diff
        return state_dict
