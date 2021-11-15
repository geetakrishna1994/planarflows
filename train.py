from torch import torch,optim
import numpy as np
import planarflow
from flow import Flow
from loss import Loss
import distributions
import matplotlib.pyplot as plt
from utils import *


epochs = 2e3
lr = 1e-2
sample_size = 500
dim = 2
report_steps = 500


def train(target_dist,flow_length) :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_density = distributions.target_density(target_dist)
    base_density = distributions.base_distribution(dim = dim,device = device)
    flow = Flow(n_layers=flow_length)
    flow = flow.to(device)
    kl = Loss(base_density,target_density)
    optimizer = optim.Adam(flow.parameters(),lr = lr)

    count = 0
    for epoch in range(1,int(epochs)+1):
        count += 1
        optimizer.zero_grad()
        z0 = base_density.sample((sample_size,)).to(device)
        zk,total_log_jacobian = flow(z0)
        if torch.isnan(total_log_jacobian).sum() > 0:
            print('Log Det jacobian is null','epochs run : ',epoch)
            break

        loss = kl.forward(zk,z0,total_log_jacobian)

        loss.backward()
        optimizer.step()

        if epoch%report_steps == 0:
            print("epoch {:05d}/{},kl : {}".format(epoch,int(epochs),loss))



    
    x = y = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(x, y)
    shape = X.shape
    X_flatten, Y_flatten = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))
    Z = torch.from_numpy(np.concatenate([X_flatten, Y_flatten], 1)).to(torch.float)
    
    plt.figure(figsize=(16,5))

    density = distributions.target_density(target_dist)
    U = torch.exp(-density(Z))
    U = U.reshape(shape)
    plt.subplot(1,2,1)
    plt.pcolormesh(X, Y, U, cmap="inferno", rasterized=True)

    z0 = base_density.sample((1000,)).to(device)
    zK,_ = flow(z0)
    zK = zK.cpu().detach().numpy()
    plt.subplot(1,2,2)
    plt.scatter(zK[:,0], zK[:,1])
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.savefig(target_dist+'_'+str(flow_length) +'.png')
    

for length in [32] :
    train('U1',flow_length=length)
    train('U2',flow_length=length)
    train('U3',flow_length=length)
    train('U4',flow_length=length)
