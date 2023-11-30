import torch 
from torch import nn
import torch.quantization as qat
from torch.quantization import QuantStub, DeQuantStub
import torch.nn.functional as F

def stochastic_round(t):
    """
    Perform stochastic rounding to nearest integer on a PyTorch tensor.
    Numbers are rounded to the nearest integer with probabilities proportional 
    to their proximity to the integers.
    """
    floor = torch.floor(t)
    ceil = torch.ceil(t)
    return torch.where(torch.rand_like(t) < (t - floor), floor, ceil).to(torch.int)
    # return torch.round(t).to(torch.int)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.l1 = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(dim_hidden, dim_out)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x
    
    def reset(self):
        nn.init.uniform_(self.l1.weight, -torch.sqrt(torch.tensor(1/self.dim_in)), torch.sqrt(torch.tensor(1/self.dim_in)))
        nn.init.uniform_(self.l2.weight, -torch.sqrt(torch.tensor(1/self.dim_hidden)), torch.sqrt(torch.tensor(1/self.dim_hidden)))
#         nn.init.uniform_(self.l1.bias, -r/10, r/10)
#         nn.init.uniform_(self.l2.bias, -r/10, r/10)
    
    def parameter_range(self):
        par_dict = self.state_dict()
        w1_min = par_dict["l1.weight"].min().item()
        w1_max = par_dict["l1.weight"].max().item()
        w2_min = par_dict["l2.weight"].min().item()
        w2_max = par_dict["l2.weight"].max().item()        
        b1_min = par_dict["l1.bias"].min().item()
        b1_max = par_dict["l1.bias"].max().item()
        b2_min = par_dict["l2.bias"].min().item()
        b2_max = par_dict["l2.bias"].max().item() 
        return [[w1_min, w1_max], [w2_min, w2_max], [b1_min, b1_max], [b2_min, b2_max]]    
    
    def quantize(self):
        par_dict = self.state_dict()
        qm = QT(self.dim_in, self.dim_hidden, self.dim_out)
        qm.w1 = torch.clamp(stochastic_round(par_dict["l1.weight"] / qm.s_w1), -127, 127).to(torch.int).t()
        qm.w2 = torch.clamp(stochastic_round(par_dict["l2.weight"] / qm.s_w2), -127, 127).to(torch.int).t()
        qm.b1 = stochastic_round(par_dict["l1.bias"] / qm.s_b1)
        qm.b2 = stochastic_round(par_dict["l2.bias"] / qm.s_b2)
        return qm


class QT:
    def __init__(self, dim_in, dim_hidden, dim_out, w1_r = 1.0, w2_r = 1.5, y_r = 5.0):
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.w1 = torch.zeros((dim_in, dim_hidden)).to(torch.int)
        self.w2 = torch.zeros((dim_hidden, dim_out)).to(torch.int)
        self.b1 = torch.zeros(dim_hidden).to(torch.int)
        self.b2 = torch.zeros(dim_out).to(torch.int)
        
        self.s_x = 1.0/255
        y_r = y_r if y_r is not None else 5.0
        w1_r = w1_r if w1_r is not None else 1.0
        w2_r = w2_r if w2_r is not None else 1.5
        self.s_y = y_r / 255.0
        self.s_w1 = w1_r / 127.0
        self.s_w2 = w2_r / 127.0
        self.s_b1 = self.s_x * self.s_w1
        self.s_b2 = self.s_y * self.s_w2

        self.scale_dict = {
            "l1.weight": self.s_w1,
            "l2.weight": self.s_w2,
            "l1.bias": self.s_b1,
            "l2.bias": self.s_b2
        }

        self.b2_grad_s = 1.0/32767.0
        self.w2_grad_s = self.b2_grad_s * self.s_y

        self.b1_grad_s = self.s_w2 * self.b2_grad_s
        self.w1_grad_s = self.b1_grad_s * self.s_x
    
    def forward(self, x):
        x = F.relu(x * self.w1 + self.b1)
        x = torch.clamp(x * self.s_b1/self.s_y, min=0, max=255).to(torch.int)  
        return F.softmax((x * self.w2 + self.b2) * self.model.s_b2, dim=-1)
    
    def reset(self):
        self.w1 = torch.round((torch.rand(self.w1.shape) - 0.5) * 2 *torch.sqrt(torch.tensor(1/self.dim_in)) /self.s_w1).to(torch.int)
        self.w2 = torch.round((torch.rand(self.w2.shape) - 0.5) * 2 * torch.sqrt(torch.tensor(1/self.dim_hidden))/self.s_w2).to(torch.int)
        self.b1 = torch.round((torch.rand(self.b1.shape) - 0.5) * 2 *(torch.tensor(1/self.dim_hidden)/self.s_b1)).to(torch.int)
        self.b2 = torch.round((torch.rand(self.b2.shape) - 0.5) * 2 *(torch.tensor(1/self.dim_out)/self.s_b2 )).to(torch.int)

    def load_state_dict(self, target_dict):
        self.w1 = target_dict['l1.weight'].clone()
        self.w2 = target_dict['l2.weight'].clone()
        self.b1 = target_dict['l1.bias'].clone()
        self.b2 = target_dict['l2.bias'].clone()

    def state_dict(self):
        return  {
            'l1.weight': self.w1.clone(),
            'l2.weight': self.w2.clone(),
            'l1.bias': self.b1.clone(),
            'l2.bias': self.b2.clone()
        }

    def parameter_range(self):
        w1_min = self.w1.min().item()
        w1_max = self.w1.max().item()
        w2_min = self.w2.min().item()
        w2_max = self.w2.max().item()        
        b1_min = self.b1.min().item()
        b1_max = self.b1.max().item()
        b2_min = self.b2.min().item()
        b2_max = self.b2.max().item() 
        return [[w1_min* self.s_w1, w1_max* self.s_w1], 
                [w2_min* self.s_w2, w2_max* self.s_w2], 
                [b1_min* self.s_b1, b1_max *self.s_b1], 
                [b2_min* self.s_b2, b2_max* self.s_b2]]    

    def dequantize(self):
        m = MLP(self.dim_in, self.dim_hidden, self.dim_out)
        state_dict = m.state_dict()
        state_dict['l1.weight'] = (self.w1 * self.s_w1).to(torch.float)
        state_dict['l2.weight'] = (self.w2 * self.s_w2).to(torch.float)
        state_dict['l1.bias'] = (self.b1 * self.s_b1).to(torch.float)   
        state_dict['l2.bias'] = (self.b2 * self.s_b2).to(torch.float)   
        m.load_state_dict(state_dict)
        return m
