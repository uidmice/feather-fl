import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from  model import MLP, QT, stochastic_round



class Client:
    def __init__(self, dim_in, dim_hidden, dim_out, dataset, mode = 'fp'):
        self.mode = mode   # FP: FLOATING POINT, qt: QUANTIZATION AWARE TRAINING
        self.dataset = dataset
 
        if mode == 'fp':
            self.model = MLP(dim_in, dim_hidden, dim_out)
            self.global_model = MLP(dim_in, dim_hidden, dim_out)
            
        else: #qt
            self.model = QT(dim_in, dim_hidden, dim_out)
            self.global_model = QT(dim_in, dim_hidden, dim_out)
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
    
    def local_SGD_GT(self, batch_size, steps, lr):
        data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        tot_loss = 0 
        for j in range(round(steps * batch_size/len(self.dataset))):
            for i, data in enumerate(data_loader):
                inputs = data[0]
                labels = data[1]
                inputs = stochastic_round((inputs * 255)).to(torch.float)
                # bs x hidden_dim
                y = F.relu(torch.matmul(inputs , self.model.w1.to(torch.float)) + self.model.b1.to(torch.float))
                y = torch.clamp(y * self.model.s_b1/self.model.s_y, min=0, max=255)
                y = stochastic_round(y).to(torch.float)

                # bs x output_dim
                output = F.softmax((torch.matmul(y , self.model.w2.to(torch.float)) + self.model.b2.to(torch.float)) * self.model.s_b2, dim=-1)
                with torch.no_grad():
                    loss = self.loss(output, labels)
                    tot_loss += loss.item()
                #bs x output_dim
                dl = stochastic_round((output - F.one_hot(labels, 2)) / self.model.b2_grad_s).to(torch.float)

                # (output, )
                b2_grad = torch.sum(dl, dim=0)

                # (hidden_dim  x output_dim)
                w2_grad = torch.matmul(y.t(), dl)

                # bs x hidden_dim
                dy = torch.matmul(dl, self.model.w2.to(torch.float).t() )
                dy = torch.where(y > 0, dy, torch.tensor(0))

                # (hidden_dim, )
                b1_grad = torch.sum(dy, dim=0)
                # (input_dim  x hidden_dim)
                w1_grad = torch.matmul(inputs.t().to(torch.float), dy)

                w1 = self.model.w1.to(torch.float) + w1_grad * self.model.w1_grad_s * lr / len(inputs) / self.model.s_w1
                w2 = self.model.w2.to(torch.float) + w2_grad * self.model.w2_grad_s * lr / len(inputs) / self.model.s_w2
                b1 = self.model.b1.to(torch.float) + b1_grad * self.model.b1_grad_s * lr / len(inputs) / self.model.s_b1
                b2 = self.model.b2.to(torch.float) + b2_grad * self.model.b2_grad_s * lr / len(inputs) / self.model.s_b2

                # stochastic rounding
                self.model.w1 = torch.clamp(stochastic_round(w1), -127, 127).to(torch.int)
                self.model.w2 = torch.clamp(stochastic_round(w2), -127, 127).to(torch.int)
                self.model.b1 = stochastic_round(b1)
                self.model.b2 = stochastic_round(b2)

        return tot_loss
    def test(self):
        if self.mode == 'fp':
            qm = self.model.quantize()
            data = self.dataset[0]
            y = F.relu(self.model.l1(data[0]))
            q = y /qm.s_y
            qy = F.relu(torch.matmul((data[0] * 255).to(torch.int) , qm.w1) + qm.b1)
            qy = torch.clamp(qy * qm.s_b1/qm.s_y, min=0, max=255).to(torch.int)
            qa = F.softmax((torch.matmul(qy , qm.w2) + qm.b2) * qm.s_b2, dim=-1)


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
