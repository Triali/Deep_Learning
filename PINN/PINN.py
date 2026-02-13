import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self,nin,nout,n_layers, width):
        layers = []
        layers.append(nn.Linear(nin,width))
        layers.append(nn.Tanh())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(width,width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width,nout))
        layers.append(nn.Tanh())
        self.network=nn.Sequential(*layers)

    def forward(self,inputs):
        return self.network(inputs)

# def residual(model,t,params):
#     m,mu,k = params
#     y=model(t)
#     dydt = torch.autograd.grad(y,t,grad_outputs=torch.ones_like(y),create_graph=True)[0]
    
#     d2ydt2 = torch.autograd.grad(dydt,t,grad_outputs=torch.ones_like(y),create_graph=True)[0]
#     return m*d2ydt2+mu*dydt+k*y

def residual(model,inputs,params):
    x,t = inputs
    u=model(x,t)
    dudt = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(y),create_graph=True)[0]
    dudx = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(y),create_graph=True)[0]
    d2udx2 = torch.autograd.grad(dudx,x,grad_outputs=torch.ones_like(y),create_graph=True)[0]
    
    return dudt+u*dudx-.01/torch.pi*d2udx2


def boundaries(model,x,t):
    u = model(x,t)
    if x==1 or x==-1:
        return u
    if t==0:
        return -1*torch.sin(torch.pi*x)
    return 0

def create_boundaries(bound_size):
    x = torch.linspace(-1,bound_size,1).view(-1,1)
    zeros = torch.zeros_like(x)
    t1 = torch.ones_like(x)
    t2 = torch.full_like(x,-1)
    pnts1 = torch.cat([zeros,x],dim=1)
    pnts2 = torch.cat([t1,zeros],dim=1)
    pnts3 = torch.cat([t1,zeros],dim=1)
    boundaries = torch.cat([pnts1, pnts2, pnts3], dim=0)
    boundaries.requires_grad_(True)
    return boundaries

def create_collocation(nx,nt):
    x = torch.linspace(-1,1,nx)
    t = torch.linspace(0,1,nt)

pinn = MLP(2,1,9,20)
bound_size = 10
t_bound = create_boundaries(bound_size)

