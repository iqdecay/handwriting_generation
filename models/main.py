# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distrib
import numpy as np



class Unconditional_LSTM(nn.Module):

    def __init__(self, input_size, hidden_dim, output_dim):
        super(Unconditional_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        predictions = torch.tensor(()).new_zeros(sequence.shape)
        for k, char in enumerate(sequence):
            char_unsqueezed = char.unsqueeze(0)
            h, c = self.lstm(char_unsqueezed)
            linear_output = self.out(h)
            mixture_output = self.mixture(linear_output)
            predictions[k] = mixture_output
        return predictions

    def mixture(self, output):
        e_est, mu_x, mu_y, std_x_est, std_y_est, rho_est, pi_est = output.squeeze().requires_grad_(True)

        # From estimated to real value
        e = torch.sigmoid(e_est).requires_grad_(True)
        rho = torch.tanh(rho_est).requires_grad_(True)
        std_x = torch.exp(std_x_est).requires_grad_(True)
        std_y = torch.exp(std_y_est).requires_grad_(True)
        cov = (rho * std_x * std_y).requires_grad_(True)

        # Parameters of the bivariate Gaussian
        cov_matrix = torch.tensor([[std_x ** 2, cov], [cov, std_y ** 2]]).requires_grad_(True)
        means = torch.tensor([mu_x, mu_y]).requires_grad_(True)

        #  Instantiate distributions
        bivariate_distrib = distrib.MultivariateNormal(means, cov_matrix)
        bernoulli_distrib = distrib.Bernoulli(torch.tensor([e]))

        # Sample from distributions
        x, y = bivariate_distrib.sample().requires_grad_(True)
        end_of_stroke = bernoulli_distrib.sample().requires_grad_(True)
        return torch.tensor([end_of_stroke, x, y]).requires_grad_(True)


def train_model():
    filepath = '../data/strokes-py3.npy'
    strokes_data = np.load(filepath, allow_pickle=True)

    # Create a train / test set
    n_sample = len(strokes_data)
    partition_index = int(n_sample * 0.8)
    X_train, X_test = strokes_data[:partition_index], strokes_data[partition_index:]

    parameters = {
        "input_size": 3,
        "output_dim": 7,  # assume M = 1
        "hidden_dim": 20
    }

    n_epochs = 10
    model = Unconditional_LSTM(**parameters)
    loss_function = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters())
    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch+1}/{n_epochs}")
        for k, strokes in enumerate(X_train):
            if (k + 1) % 100 == 0:
                print(f"Epoch {epoch+1} sample {k+1}")
            model.zero_grad()
            strokes_in = torch.tensor(strokes)
            out = model(strokes_in)
            loss = loss_function(strokes_in, out)
            loss.backward()
            optimizer.step()
    return model


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