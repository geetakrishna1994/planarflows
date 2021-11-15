import torch

class Loss :
    def __init__(self, base,target) :
        self.base = base
        self.target = target

    def forward(self,zk,z0,total_log_jacobian) :
        base = self.base.log_prob(z0)
        # print("here")
        log_learned = base - total_log_jacobian
        log_target = -self.target(zk).view(-1) 
        # print(log_learned.device, log_target.device)
        loss = torch.mean(log_learned - log_target)
        return loss
