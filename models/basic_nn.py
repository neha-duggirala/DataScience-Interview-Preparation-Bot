import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

from utils import create_line_plot, plot_losses


class BasicNN(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.w01 = nn.Parameter(torch.tensor(0.56), requires_grad=False)
        self.b01 = nn.Parameter(torch.tensor(0.23), requires_grad=False)
        self.w02 = nn.Parameter(torch.tensor(-0.16), requires_grad=False)
        self.b02 = nn.Parameter(torch.tensor(0.41), requires_grad=False)

        self.w11 = nn.Parameter(torch.tensor(-0.003), requires_grad=False)
        self.w12 = nn.Parameter(torch.tensor(-0.75), requires_grad=False)

        self.b_final = nn.Parameter(torch.tensor(0.39), requires_grad=False)

    def forward(self, ip):

        upper_first = ip * self.w01 + self.b01
        activation_first = F.relu(upper_first)
        scaled_first = activation_first * self.w11

        lower_first = ip * self.w02 + self.b01
        activation_second = F.relu(lower_first)
        scaled_second = activation_second * self.w12

        op = F.relu(scaled_first + scaled_second + self.b_final)

        return op


class BasicNN_train(nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.w01 = nn.Parameter(torch.tensor(0.56), requires_grad=True)
        self.b01 = nn.Parameter(torch.tensor(0.23), requires_grad=True)
        self.w02 = nn.Parameter(torch.tensor(0.16), requires_grad=True)
        self.b02 = nn.Parameter(torch.tensor(0.41), requires_grad=True)

        self.w11 = nn.Parameter(torch.tensor(0.003), requires_grad=True)
        self.w12 = nn.Parameter(torch.tensor(0.75), requires_grad=True)

        self.b_final = nn.Parameter(torch.tensor(0.), requires_grad=True)


    def forward(self, ip):

        upper_first = ip * self.w01 + self.b01
        activation_first = F.relu(upper_first)
        scaled_first = activation_first * self.w11

        lower_first = ip * self.w02 + self.b01
        activation_second = F.relu(lower_first)
        scaled_second = activation_second * self.w12

        op = F.relu(scaled_first + scaled_second + self.b_final)

        return op

def train(model, ips, ops):
    
    optimizer = Adam(model.parameters(), lr= 0.1)

    print(f"before optimization, {model.b_final}")
    losses = []
    for epoch in range(1000):
        total_loss = 0

        for ip, op in zip(ips,ops):
            predicted_op = model(ip)
            loss = (op - predicted_op)**2

            loss.backward()

            total_loss+=float(loss)

        losses.append(total_loss)
        if (total_loss < 0.0001):
            print("Num steps: " + str(epoch))
            break
        
        optimizer.step() ## take a step toward the optimal value.
        optimizer.zero_grad()

    plot_losses(losses)
    

    






if __name__ == "__main__":
    # inference
    inference_input = torch.linspace(start=0, end=1, steps=11)
    model = BasicNN()
    output_values = model(inference_input)
    print(output_values)
    create_line_plot(inference_input, output_values)

    # datapoints
    inputs = torch.tensor([0, 0.5, 1])
    outputs = torch.tensor([0, 1, 0])
    model = BasicNN_train()
    model_ops = model(inputs)
    train(model, inputs, outputs)
    create_line_plot(inputs, model_ops.detach())
