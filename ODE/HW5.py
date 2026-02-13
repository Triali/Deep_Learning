import torch
from torch import nn
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import os

class NeuralODE(nn.Module):
    def __init__(self,nin, nout,layers):
        super(NeuralODE,self).__init__()
        m_layers = []
        m_layers.append(nn.Linear(nin,layers[0]))
        # print(f'[{nin},{layers[0]}]')
        m_layers.append(nn.SiLU())
        for i in range(len(layers)-1):
            m_layers.append(nn.Linear(layers[i],layers[i+1]))
            # print(f'[{layers[i]},{layers[i+1]}]')
            m_layers.append(nn.SiLU())
        m_layers.append(nn.Linear(layers[-1],nout))
        # print(f'[{layers[-1]},{nout}]')
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
    data1 = pd.read_csv('ODE/DailyDelhiClimateTrain.csv')
    data2 = pd.read_csv('ODE/DailyDelhiClimateTest.csv')

    # print(data1.shape)
    # print(data2.shape)

    data_pd = pd.concat([data1,data2],ignore_index=True)
    # print(data_pd.shape)
    data_pd['date'] = pd.to_datetime(data_pd['date'])
    
    monthly_avg = (
        data_pd
        .set_index('date')
        .resample('MS')     # calendar month
        .mean()
        .reset_index()
    )    
    
    
    print(monthly_avg.head())

    data = monthly_avg.to_numpy()[:,1:]
    data = np.asarray(data, dtype=np.float32)
    
    mean = data.mean(axis=0)
    std  = data.std(axis=0)

    data_norm = (data - mean) / std
    print(data_norm.shape)

    


    t_train = torch.from_numpy(np.arange(48).astype(np.float32)) #[nt,]
    y_train = torch.from_numpy(data_norm[:48, :].astype(np.float32)) #[nt,2]
    print(t_train.shape)
    print(y_train.shape)

    t_test = torch.from_numpy(np.arange(48,52).astype(np.float32)) #[nt,]
    y_test = torch.from_numpy(data_norm[48:, :].astype(np.float32)) #[nt,2]
    print(t_test.shape)
    print(y_test.shape)

    model = NeuralODE(4,4,[15,20,20,20,15])
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    lossfn = nn.MSELoss()
    plt.ion()

    epochs = 10000
    losses = np.zeros(epochs)

    fig_loss, ax_loss = plt.subplots(figsize=(7,4))
    loss_line, = ax_loss.plot([], [], lw=2)
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE Loss")

    fig_preds, axs = plt.subplots(4,1,figsize=(8, 10), sharex=True)

    labels = ["Temperature", "Humidity", "Wind Speed", "Pressure"]
    colors = ["r", "b", "g", "k"]
    pred_lines = []

    for i in range(4):
        # predicted line
        pred_line, = axs[i].plot([], [], color=colors[i], lw=2, label=f"Predicted {labels[i]}")
        # true line
        axs[i].plot(t_train, y_train[:, i], color=colors[i], linestyle="--", label=f"True {labels[i]}")
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
        pred_lines.append(pred_line)

    axs[-1].set_xlabel("t")
    plt.show(block=False)

    for e in range(epochs):
        if e < 500:
            ysub = y_train[:4, :]
            tsub = t_train[:4]
        elif e < 1000:
            ysub = y_train[:8, :]
            tsub = t_train[:8]
        elif e<1500:
            ysub = y_train[:12, :]
            tsub = t_train[:12]
        elif e<2000:
            ysub = y_train[:16, :]
            tsub = t_train[:16]
        elif e<2500:
            ysub = y_train[:20, :]
            tsub = t_train[:20]
        elif e<3000:
            ysub = y_train[:24, :]
            tsub = t_train[:24]
        elif e<3500:
            ysub = y_train[:28, :]
            tsub = t_train[:28]
        elif e<4000:
            ysub = y_train[:32, :]
            tsub = t_train[:32]
        elif e<4500:
            ysub = y_train[:36, :]
            tsub = t_train[:36]
        elif e<5000:
            ysub = y_train[:40, :]
            tsub = t_train[:40]
        elif e<5500:
            ysub = y_train[:44, :]
            tsub = t_train[:44]
        elif e<6000:
            ysub = y_train[:48, :]
            tsub = t_train[:48]
        else:
            ysub = y_train
            tsub = t_train
        print(e,losses[e])
        losses[e] = train(ysub, tsub, model, optimizer, lossfn)
    #     print(losses[e])

        model.eval()
        with torch.no_grad():
            yhat = model(y_train[0, :], t_train)
        if e % 5 == 0:
        # update loss
            loss_line.set_data(range(e+1), losses[:e+1])
            ax_loss.relim()
            ax_loss.autoscale_view()

            # update predictions
            for i in range(4):
                pred_lines[i].set_data(t_train, yhat[:, i])
                axs[i].relim()
                axs[i].autoscale_view()

            plt.pause(0.01)
    torch.save(model,'/ODE/NNODE_weights.pth')
    print('DONE')
    plt.ioff()
    plt.show()


        