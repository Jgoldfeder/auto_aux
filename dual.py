import torch
import torch.nn as nn
import numpy as np

class DenseWrapper(nn.Module):
    def __init__(self, model):
        super(DenseWrapper, self).__init__()
        self.model = model        
    def forward(self,x):
        return self.model(x,True)[1]
        

class DualModel(nn.Module):
    def __init__(self, model,num_classes,n_ways=1):

        super(DualModel, self).__init__()
        self.default_cfg = model.default_cfg
        self.num_classes = num_classes
        

   
        self.model = model
        self.fc = model.fc
        model.fc = nn.Identity()
        
        self.dense_fc = nn.Linear(self.fc.in_features,4096)
        self.relu = nn.LeakyReLU()

        self.pre_fc = nn.Linear(self.fc.in_features,64)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # state size. 64 x 4 x 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # state size. 32 x 8 x 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # state size. 16 x 16 x 16
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # state size. 8 x 32 x 32
            nn.ConvTranspose2d(8, 1, 4, 2, 1)
        )
        self.sig = nn.Sigmoid()
        self.post_fc = nn.Linear(4096,4096*n_ways)

        self.taskmodules = nn.ModuleList([self.fc, self.dense_fc,self.deconvs,self.pre_fc,self.post_fc])

    def forward(self,x,on=False):

        x = self.model(x)

        if on:
            x1 =  self.fc(x)
            #x2 = self.sig(self.dense_fc(x))
            x2 = self.post_fc(self.deconvs(self.pre_fc(x).unsqueeze(-1).unsqueeze(-1)).squeeze().squeeze().reshape(-1,4096))
            return x1, x2
        return self.fc(x)


class DualModelVIT(nn.Module):
    def __init__(self, model,num_classes,n_ways=1):

        super(DualModelVIT, self).__init__()
        self.default_cfg = model.default_cfg
        self.num_classes = num_classes
        

        # code for tresnet
        self.fc = model.head
        print(self.fc)
        model.head = nn.Identity()    
        self.model =model
        
        #self.model = model
        #self.fc = nn.Linear(1000,196 )

        self.dense_fc = nn.Linear(self.fc.in_features,4096*n_ways)

        self.relu = nn.LeakyReLU()

        self.pre_fc = nn.Linear(self.fc.in_features,64)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            # state size. 64 x 4 x 4
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # state size. 32 x 8 x 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # state size. 16 x 16 x 16
            nn.ConvTranspose2d(16, 8, 4, 2, 1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            # state size. 8 x 32 x 32
            nn.ConvTranspose2d(8, 1, 4, 2, 1)
        )
        self.sig = nn.Sigmoid()
        self.post_fc = nn.Linear(4096,4096)

        self.taskmodules = nn.ModuleList([self.fc, self.dense_fc,self.deconvs,self.pre_fc,self.post_fc])

    def forward(self,x,on=False):

        x = self.model(x)


        if on:
            x1 =  self.fc(x)
            x2 = nn.Flatten(1)(self.dense_fc(x))
            #x2 = self.sig(self.post_fc(self.deconvs(nn.Flatten(1)(self.pre_fc(x)).unsqueeze(-1).unsqueeze(-1)).squeeze().squeeze().reshape(-1,4096)))
            return x1, x2
        return self.fc(x)




class DualLoss(nn.Module):
    """This is label smoothing loss function.
    """
    def __init__(self,loss,weights,num_classes):
        super(DualLoss, self).__init__()
        self.dense_loss =  nn.BCEWithLogitsLoss()
        self.categorical_loss = loss
        self.dense_labels = torch.tensor(np.random.choice([0, 1], size=(num_classes,64*64)).astype("float32"))
        self.emb = nn.Embedding.from_pretrained(self.dense_labels)
        self.weights = weights

    def forward(self,output,target,seperate=False):
        # dense_target = []
        # for t in target:
        #     dense_target.append(self.dense_labels[t])
        # dense_target = torch.stack(dense_target).cuda()

        dense_target = self.emb(target)
        loss1 = self.categorical_loss(output[0],target)*self.weights[0]
        loss2 = self.dense_loss(output[1],dense_target)*self.weights[1]
        if seperate:
            return [loss1,loss2]
        return loss1 + loss2

class DualLossSmooth(nn.Module):
    """This is label smoothing loss function.
    """
    def __init__(self,loss,weights,num_classes):
        super(DualLossSmooth, self).__init__()
        self.dense_loss =  nn.MSELoss()
        self.categorical_loss = loss
        self.dense_labels = torch.tensor(np.random.random_sample( size=(num_classes,64*64)).astype("float32")*80-40)
        self.emb = nn.Embedding.from_pretrained(self.dense_labels)
        self.weights = weights

    def forward(self,output,target,seperate=False):
        # dense_target = []
        # for t in target:
        #     dense_target.append(self.dense_labels[t])
        # dense_target = torch.stack(dense_target).cuda()

        dense_target = self.emb(target)
        loss1 = self.categorical_loss(output[0],target)*self.weights[0]
        loss2 = self.dense_loss(output[1],dense_target)*self.weights[1]
        if seperate:
            return [loss1,loss2]
        return loss1 + loss2





class NWayLoss(nn.Module):
    """This is label smoothing loss function.
    """
    def __init__(self,loss,weights,num_classes,n):
        super(NWayLoss, self).__init__()
        self.dense_loss =  nn.CrossEntropyLoss()
        self.categorical_loss = loss
        choices = [x for x in range(n)]
        self.dense_labels = torch.tensor(np.random.choice(choices, size=(num_classes,64*64))   )#.astype("float32"))
        self.n=n
        self.weights = weights

    def forward(self,output,target,seperate=False):
        dense_target = []
        for t in target:
            dense_target.append(self.dense_labels[t])
        dense_target = torch.stack(dense_target).cuda()
        #print(dense_target.shape,output[1].shape)
        loss1 = self.categorical_loss(output[0],target)*self.weights[0]
        dense_out = output[1].reshape(-1,4096,self.n)
        batch = dense_out.shape[0]
        loss2 = self.dense_loss(dense_out.reshape(batch*4096,self.n),dense_target.reshape(batch*4096))*self.weights[1]

        if seperate:
            return [loss1,loss2]
        return loss1 + loss2




 # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.




import math
import torch
from torch.optim.optimizer import Optimizer
import time
import numpy as np





class MetaBalance(Optimizer):
    r"""Implements MetaBalance algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        relax factor: the hyper-parameter to control the magnitude proximity
        beta: the hyper-parameter to control the moving averages of magnitudes, set as 0.9 empirically
    """
    def __init__(self, params, relax_factor=0.4, beta=0.9):
        if not 0.0 <= relax_factor < 1.0:
            raise ValueError("Invalid relax factor: {}".format(relax_factor))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta: {}".format(beta))
        defaults = dict(relax_factor=relax_factor, beta=beta)
        super(MetaBalance, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, loss_array):#, closure=None
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()

        self.balance_GradMagnitudes(loss_array)

        #return loss

    def balance_GradMagnitudes(self, loss_array):

      for loss_index, loss in enumerate(loss_array):
        loss.backward(retain_graph=True)
        for group in self.param_groups:
          for p in group['params']:

            if p.grad is None:
              print("breaking")
              break
            if p.grad.is_sparse:
              raise RuntimeError('MetaBalance does not support sparse gradients')
            state = self.state[p]

            # State initialization
            if len(state) == 0:
              for j, _ in enumerate(loss_array):
                if j == 0: p.norms = [torch.zeros(1).cuda()]
                else: p.norms.append(torch.zeros(1).cuda())

            # calculate moving averages of gradient magnitudes
            beta = group['beta']
            p.norms[loss_index] = (p.norms[loss_index]*beta) + ((1-beta)*torch.norm(p.grad))

            # narrow the magnitude gap between the main gradient and each auxilary gradient
            relax_factor = group['relax_factor']
            p.grad = (p.norms[0] * p.grad/ p.norms[loss_index]) * relax_factor + p.grad * (1.0 - relax_factor)

            if loss_index == 0:
              state['sum_gradient'] = torch.zeros_like(p.data)
              state['sum_gradient'] += p.grad
            else:
              state['sum_gradient'] += p.grad

            # have to empty p.grad, otherwise the gradient will be accumulated
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

            if loss_index==len(loss_array) - 1:

              p.grad = state['sum_gradient']       