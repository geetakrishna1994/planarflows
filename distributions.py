import torch
import numpy as np

def w1(z):
    z1,_ = z[:,0],z[:,1]
    return torch.sin((2*np.pi*z1)/4)

def w2(z):
    z1,_ = z[:,0],z[:,1]
    return 3*torch.exp(-0.5*((z1-1)/0.6)**2)

def w3(z):
    z1,_ = z[:,0],z[:,1]
    x = (z1-1)/0.3
    return 3/(1+torch.exp(-x))

def U1(z):
    # print(z.shape)
    z1,_ = z[:,0],z[:,1]
    term1 = ((torch.norm(z,p=2,dim=1)-2)/0.4)**2
    term2 = torch.exp(-0.5*((z1-2)/0.6)**2)
    term3 = torch.exp(-0.5*((z1+2)/0.6)**2) 
    return (0.5*term1-torch.log(term2+term3))
    
def U2(z):
    z1,z2 = z[:,0],z[:,1]
    return 0.5*((z2-w1(z))/0.4)**2

def U3(z):
    z1,z2 = z[:,0],z[:,1]
    term1 = torch.exp(-0.5*((z2-w1(z))/0.35)**2)
    term2 = torch.exp(-0.5*((z2-w1(z)+w2(z))/0.35)**2)
    return -torch.log(term1+term2+1e-4)

def U4(z):
    z1,z2 = z[:,0],z[:,1]
    term1 = torch.exp(-0.5*((z2-w1(z))/0.4)**2)
    term2 = torch.exp(-0.5*((z2-w1(z)+w3(z))/0.35)**2)
    return -torch.log(term1+term2)

def target_density(name):
	if name == 'U1':
		return U1
	elif name == 'U2':
		return U2
	elif name == 'U3':
		return U3
	elif name == 'U4':
		return U4

def base_distribution(dim, device):
	if dim == 1:
		return torch.distributions.normal.Normal(torch.tensor(0.).to(device),torch.tensor(1.).to(device))
	elif dim == 2:
		return torch.distributions.MultivariateNormal(
            torch.zeros(2).to(device), torch.eye(2).to(device))