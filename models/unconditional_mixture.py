# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distrib
import numpy as np


class Unconditional_LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_dim, output_dim, M):
        super(Unconditional_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.M = M

    def forward(self, sequence):
        predictions = torch.tensor(()).new_zeros(se)
        for k, char in enumerate(sequence) :
            char_unsqueezed = char.unsqueeze(0)
            h, c = self.lstm(char_unsqueezed)
            linear_output =  self.out(h)
            e, mu_x, mu_y, std_x, std_y, rho, pi = self.mixture(linear_output)
            output = self.probability(e, mu_x, mu_y, std_x, std_y, rho, pi, char)
            predictions[k] = output
        return predictions
          
    

    def Normal(self, mu_x, mu_y, std_x, std_y, rho, x, y):
        Z1 = ((x - mu_x)/std_x)**2 + ((y -mu_y)/std_y)**2
        Z2 = 2*rho*(x-mu_x)*(y - mu_y)/(std_x * std_y)
        Z = Z1 - Z2
        inv_cte = (2 * np.pi * std_x * std_y * torch.sqrt(1 - rho*rho))
        cte = 1/inv_cte
        argument = -Z /(2*(1-rho*rho)) 
        return cte * torch.exp(argument).requires_grad_(True)


    def probability(self, e, mu_x, mu_y, std_x, std_y, rho, pi, char):
      end_of_stroke, x, y = char
      pr_x_y = 0
      for k in range(self.M):
        xmean, ymean, x_std, y_std = mu_x[k], mu_y[k], std_x[k], std_y[k]
        corr = rho[k]
        pr_x_y += pi[k] * self.Normal(xmean, ymean, x_std, y_std, corr, x, y)
      if end_of_stroke == 1 :
        return pr_x_y * e
      else :
        return pr_x_y * (1-e)


    def mixture(self, output):
        M = self.M
        output = output.squeeze().requires_grad_(True)
        e_est = output[0].requires_grad_(True)
        parameters = output[1:].requires_grad_(True)
        mu_x = parameters[:M].requires_grad_(True)
        mu_y = parameters[M: 2*M].requires_grad_(True)
        std_x_est = parameters[2*M: 3*M].requires_grad_(True)
        std_y_est = parameters[3*M: 4*M].requires_grad_(True)
        rho_est = parameters[4*M:5*M].requires_grad_(True)
        pi_est = parameters[5*M:].requires_grad_(True)
        e = torch.sigmoid(e_est).requires_grad_(True)
        std_x = torch.exp(std_x_est).requires_grad_(True)
        std_y = torch.exp(std_y_est).requires_grad_(True)
        rho = torch.tanh(rho_est).requires_grad_(True)
        pi = nn.Softmax(0)(pi_est).requires_grad_(True)
        return e, mu_x, mu_y, std_x, std_y, rho, pi

def train_model()
    filepath = '../data/strokes-py3.npy'
    strokes_data = np.load(filepath, allow_pickle=True)

    # Create a train / test set
    n_sample = len(strokes_data)
    partition_index = int(n_sample * 0.8)
    X_train, X_test = strokes_data[:partition_index], strokes_data[partition_index:]

    parameters = {
        "input_size": 3,
        "output_dim": 13,  # assume M = 1
        "hidden_dim": 20
        "M" : 2
    }

    n_epochs = 10
    model = Unconditional_LSTM(**parameters)
    loss_function = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters())
    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch+1}/{n_epochs}")
        for k, strokes in enumerate(X_train[:10]):
            if (k + 1) % 100 == 0:
                print(f"Epoch {epoch+1} sample {k+1}")
            model.zero_grad()
            strokes_in = torch.tensor(strokes)
            out = model(strokes_in)
            loss = loss_function(strokes_in, out)
            loss.backward()
            optimizer.step()
    return model

def sample_from_model(strokes, model):
    # Take as argument a strokes, as an array (not a tensor),
    # a model, and return the output as an array (not a tensor)
    # and set the first and last "pen stroke" dimensions to 1
    # so that it can be plotted as a continuous line
    with torch.no_grad():
        strokes_in = prepare_strokes(strokes)
        strokes_in = strokes_in.squeeze()
        output = model(strokes_in.float())
        print(output.shape)
        #output = output.reshape(output.shape[1], output.shape[2])
        output[0][0] = 1
        output[-1][0] = 1
        return output.clone().detach()
def sample_from_model(model):
    """
    Return a sample pen trajectory from a model based on random input
    :param model:  Unconditional_LSTM network
    :return: numpy array representing a pen-stroke
    """
    strokes = np.random.rand(700, 3)
    with torch.no_grad():
        strokes_in = torch.tensor(strokes)
        output = model(strokes_in.float())
        return output.clone().detach()


def generate_unconditionally():
    model = train_model()
    strokes = sample_from_model(model)
    return strokes

a = generate_unconditionally()
print(a)
