import torch
from torch import nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np



class NeuralODE(nn.Module):

    def __init__(self, nin, nhidden):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(nin, nhidden),
            nn.SiLU(),  # Tanh is not the best
            nn.Linear(nhidden, nhidden),
            nn.SiLU(),
            nn.Linear(nhidden, nin)
        )

    def odefunc(self, t, y):  # provide dy/dt = f(t, y)
        return self.network(y)

    def forward(self, y0, tsteps):
        yhat = odeint(self.odefunc, y0, tsteps)
        return yhat


def train(y_train, t_train, model, optimizer, lossfn):

    model.train()
    optimizer.zero_grad()
    yhat = model(y_train[0, :], t_train)
    loss = lossfn(yhat, y_train)
    loss.backward()
    optimizer.step()

    return loss.item()


if __name__ == "__main__":

    data = np.loadtxt("in_class.txt")

    t_train = torch.tensor(data[:, 0], dtype=torch.float64)  # nt
    y_train = torch.tensor(data[:, 1:], dtype=torch.float64)  # [nt 2]


    model = NeuralODE(nin=2, nhidden=24).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.MSELoss()

    epochs = 500
    losses = np.zeros(epochs)
    for e in range(epochs):

        if e < 100:
            ysub = y_train[0:100, :]
            tsub = t_train[:100]
        else:
            ysub = y_train
            tsub = t_train

        losses[e] = train(ysub, tsub, model, optimizer, lossfn)
        print(losses[e])

        if e % 50 == 0:
            model.eval()
            with torch.no_grad():
                yhat = model(y_train[0, :], tsub)

            plt.figure()
            plt.plot(tsub, yhat[:, 0], "r")
            plt.plot(tsub, yhat[:, 1], "b")
            plt.plot(t_train, y_train[:, 0], "r--")
            plt.plot(t_train, y_train[:, 1], "b--")
            plt.show()


    model.eval()
    with torch.no_grad():
        yhat = model(y_train[0, :], t_train)

    plt.figure()
    plt.plot(range(epochs), losses)

    plt.figure()
    plt.plot(t_train, yhat[:, 0], "r")
    plt.plot(t_train, yhat[:, 1], "b")
    plt.plot(t_train, y_train[:, 0], "r--")
    plt.plot(t_train, y_train[:, 1], "b--")

    plt.show()


