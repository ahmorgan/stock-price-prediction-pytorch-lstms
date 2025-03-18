import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input_size = 1 because there is only one input feature
        # hidden_size (size of h_t vector) = 50 (adjust via trial/error)
        # num_layers = 1, >1 would mean multiple lstms
        # batch_first = true, the batch is the data which is provided first in the input tensor
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linearize_output = nn.Linear(50, 1)  # hidden state vector of size n to output of size one

    def forward(self, x, last_only=False):
        x, _ = self.lstm(x)  # pass data through network and get output vector x (h_t at final timestep)
        if last_only:
            x = x[:, -1, :]  # grab the final hidden states
        x = self.linearize_output(x)  # 8 tensors of size 50 (hidden size) now flattened to 8 arrays of 5 values each
        return x

    @staticmethod
    def train_model(model, X_train, y_train, X_test, y_test, num_epochs):
        # the adam optimizer efficiently updates model parameters based on calculated gradients
        optimizer = optim.Adam(model.parameters())  # can tune the learning rate (by def = 0.001) and betas (by def = 0.9,0.999)
        loss_function = nn.MSELoss()  # mean squared error loss, which is appropriate for single-feature output
        loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)  # loads batches of data

        for epoch in range(num_epochs):
            model.train()  # sets the model to "train mode"
            for X, y in loader:
                optimizer.zero_grad()  # set gradients to zero before backpropagation
                # this is needed because PyTorch accumulates gradients across epochs
                y_pred = model(X)  # prediction
                loss = loss_function(y_pred, y)
                loss.backward()  # computes gradients (derivatives based on loss)
                optimizer.step()  # actually update parameters using previously calculated gradients
            if epoch % 20 == 0:
                model.eval()  # switching to evaluation mode
                # disable gradient calculation with no_grad() to avoid
                # unneccesary gradient calculation during evaluation
                with torch.no_grad():
                    y_pred = model(X_test)
                    test_loss = pow(loss_function(y_pred, y_test), 0.5)  # root mean squared error, which is more easily
                    # interpretable than just mean squared error as it's in the same unit of measurement as the data
                    print(f"Loss at epoch {epoch}: {test_loss}")

        return model



