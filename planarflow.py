import numpy as np
import torch
import torch.nn as nn

class PlanarFlow(nn.Module) :
    '''
        z1 = f(z)
        f(z) = z + u * h(w.z + b)
        h(z) = tanh(z)

        to ensure that f is invertible
        u.w >= -1
        this can be achieved through following change to u
        u = u + (m(u.w) - u.w) w/||w||
        
        m(x) = -1 + log(1 + exp(x))

        paper : https://arxiv.org/pdf/1505.05770.pdf
    '''
    def __init__(self, dim=2) :
        super().__init__()
        self.w = nn.Parameter(torch.rand((1,dim)).normal_(0,0.1))
        self.u = nn.Parameter(torch.rand((1,dim)).normal_(0,0.1))
        self.b = nn.Parameter(torch.rand(()).normal_(0,0.1))
        # print(self.w.device,self.u.device)
        # print(dir(self))
    
    def h(self,z) :
        return torch.tanh(z)

    def h_prime(self, z) :
        return 1 - self.h(z)**2

    def m(self,x) :
        return -1 + torch.log(1 + torch.exp(x))

    def forward(self,z) :
        # print(self.w.device,self.u.device)
        u_dot_w = torch.mm(self.u, self.w.T)
        # ensure f is invertible
        if u_dot_w < -1 :
            self.u.data = self.u + (self.m(u_dot_w) - u_dot_w) * self.w / torch.norm(self.w)
        z_dash = torch.mm(z,self.w.T) + self.b
        z_new = z + self.u * self.h(z_dash)
        return z_new, self.logJacobian(z_dash)

    def logJacobian(self,z_dash, tol= 1e-6) :
        det = torch.abs(1 + torch.mm(self.u, self.w.T) * self.h_prime(z_dash))
        return torch.log(det + tol)

    

    





if __name__ == "__main__" :
    planar = PlanarFlow(dim = 2)
    print(planar(torch.rand((1,2))))



