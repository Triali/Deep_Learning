import torch
from torch import nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import os
class NeuralODE(nn.Module):
    def __init__(self,nin, nout,layers):
        super(NeuralODE,self).__init__()
        m_layers = []
        m_layers.append(nn.Linear(nin,layers[0]))
        print(f'[{nin},{layers[0]}]')
        m_layers.append(nn.SiLU())
        for i in range(len(layers)-1):
            m_layers.append(nn.Linear(layers[i],layers[i+1]))
            print(f'[{layers[i]},{layers[i+1]}]')
            m_layers.append(nn.SiLU())
        m_layers.append(nn.Linear(layers[-1],nout))
        print(f'[{layers[-1]},{nout}]')
        self.network=nn.Sequential(*m_layers)

    def forward(self, y0, tsteps):
        

        return odeint(self.odefunc,y0,tsteps) # [nt,2]
        
    
    def odefunc(self,t,y): #output dy/dt
        return self.network(y)

def train(y_train,t_train,model, optimizer, lossfn):

    model.train()
    optimizer.zero_grad()
    yhat = model(y_train[0, :], t_train)
    loss = lossfn(yhat,y_train)
    loss.backward()
    optimizer.step()
    
    return loss.item()

if __name__ == "__main__":

    # print(os.getcwd())
    data = np.loadtxt('ODE/in_class.txt')
    t_train = torch.tensor(data[:,0],dtype=torch.float32) #[nt,]
    y_train = torch.tensor(data[:,1:],dtype=torch.float32) #[nt,2]
    print(t_train.shape)
    print(y_train.shape)

    model = NeuralODE(2,2,[10,15,12])
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    lossfn = nn.MSELoss()
    plt.ion()

    epochs = 1000
    losses = np.zeros(epochs)

    fig1, ax1 = plt.subplots()
    loss_line, = ax1.plot([], [], lw=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")

    fig2, ax2 = plt.subplots()

    pred1, = ax2.plot([], [], "r", label="y1 pred")
    pred2, = ax2.plot([], [], "b", label="y2 pred")

    true1, = ax2.plot(t_train, y_train[:,0], "r--", label="y1 true")
    true2, = ax2.plot(t_train, y_train[:,1], "b--", label="y2 true")

    ax2.set_title("Neural ODE Solution")
    ax2.set_xlabel("t")
    ax2.legend()
    plt.show(block=False)

    for e in range(epochs):
        if e < 100:
            ysub = y_train[0:100, :]
            tsub = t_train[:100]
        elif e <200:
            ysub = y_train[0:200, :]
            tsub = t_train[:200]
        elif e <300:
            ysub = y_train[0:300, :]
            tsub = t_train[:300]
        elif e <400:
            ysub = y_train[0:400, :]
            tsub = t_train[:400]
        elif e <500:
            ysub = y_train[0:500, :]
            tsub = t_train[:500]
        elif e <600:
            ysub = y_train[0:600, :]
            tsub = t_train[:600]
        elif e <700:
            ysub = y_train[0:700, :]
            tsub = t_train[:700]
        elif e <800:
            ysub = y_train[0:800, :]
            tsub = t_train[:800]
        elif e <900:
            ysub = y_train[0:900, :]
            tsub = t_train[:900]
        else:
            ysub = y_train
            tsub = t_train

        losses[e] = train(ysub, tsub, model, optimizer, lossfn)
        print(losses[e])

        model.eval()
        with torch.no_grad():
            yhat = model(y_train[0, :], t_train)
        if e % 5 == 0:
            loss_line.set_data(range(e+1), losses[:e+1])
            ax1.relim()
            ax1.autoscale_view()

            # ----- animate solution -----
            pred1.set_data(t_train, yhat[:, 0])
            pred2.set_data(t_train, yhat[:, 1])
            ax2.relim()
            ax2.autoscale_view()
            plt.pause(0.01)
    plt.ioff()
    plt.show()

        