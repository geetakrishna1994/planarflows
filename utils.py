import numpy as np
import matplotlib.pyplot as plt
import torch
import distributions

def draw_samples(model,size=500,dim=2):
    # return zk samples and their corresponsing density value
    base = distributions.base_distribution(dim)
    z0 = base.sample((size,))
    zk,total_log_jacobian = model(z0)
    base_log_prob = base.log_prob(z0)
    final_log_prob = base_log_prob - total_log_jacobian
    qk = torch.exp(final_log_prob)
    return zk,qk

def plot_density(potential,lim=4):
    density = distributions.target_density(potential)
    x = y = np.linspace(-lim, lim, 300)
    X, Y = np.meshgrid(x, y)
    shape = X.shape
    X_flatten, Y_flatten = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))
    Z = torch.from_numpy(np.concatenate([X_flatten, Y_flatten], 1))
    U = torch.exp(-density(Z))
    U = U.reshape(shape)
    plt.pcolormesh(X, Y, U, cmap="inferno", rasterized=True)
    plt.show()