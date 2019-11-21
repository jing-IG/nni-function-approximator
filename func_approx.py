import torch
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import set_random_seed

class FuncApproxNet (torch.nn.Module):
    def __init__(self):
        super(FuncApproxNet, self).__init__()
        self.hidden_size_1 = 16
        self.hidden_size_2 = 0
        self.hidden_size_3 = 0
        self.fc1 = self._fc_block(1, self.hidden_size_1)
        last_layer_size = self.hidden_size_1
        if self.hidden_size_2 > 0:
            self.fc2 = self._fc_block(self.hidden_size_1, self.hidden_size_2)
            last_layer_size = self.hidden_size_2
        if self.hidden_size_3 > 0:
            self.fc3 = self._fc_block(self.hidden_size_2, self.hidden_size_3)
            last_layer_size = self.hidden_size_3
        self.out = self._fc_block(last_layer_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        if self.hidden_size_2:
            x = self.fc2(x)
        if self.hidden_size_3:
            x = self.fc3(x)
        x = self.out(x)
        return x

    def _fc_block(self, in_c, out_c):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_c, out_c),
            torch.nn.Sigmoid()
        )
        return block

class FuncApproxDataset(torch.utils.data.Dataset):
    def __init__(self, func_name, is_train):
        self.size = 1000 if is_train else 100
        self.is_train = is_train
        if func_name == 'quadratic':
            self.func = self.quadratic
        elif func_name == 'oscillator':
            self.func = self.oscillator
        self.prep_oscillator()

    def prep_oscillator(self):
        self.osc_x = np.arange(11) * 0.1
        self.osc_y = np.zeros_like(self.osc_x)
        y = self.osc_x[0]
        for i in range(len(self.osc_x)):
            y = -(y * y - 0.5)
            self.osc_y[i] = y

    def quadratic(self, x):
        return x*x

    def oscillator(self, x):
        y = torch.from_numpy(np.interp(x, self.osc_x, self.osc_y))
        y = y.float()
        return y

    def __getitem__(self, index):
        if self.is_train:
            data = torch.rand(1, 1)
        else:
            data = torch.empty(1, 1, dtype=torch.float)
            val = (index%self.size)/float(self.size)
            data.fill_(val) # make uniform spacing for consistent evaluation
        label = self.func(data)
        return {'x': data, 'y_exp': label}

    def __len__(self):
        return self.size

def main(func_name):
    train_ds = FuncApproxDataset(func_name, is_train=True)
    eval_ds = FuncApproxDataset(func_name, is_train=False)
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=100, shuffle=True, num_workers=2)
    net = FuncApproxNet()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    epochs = 250 if func_name == 'quadratic' else 1000
    for epoch in range(epochs):
        # log evaluation results every 5 epochs
        if epoch % 5 == 4:
            rms_error = 0
            max_error = 0
            x = []
            y = []
            y_exp = []
            net.eval()
            with torch.no_grad():
                for i in range(len(eval_ds)):
                    sample = eval_ds[i]
                    data, label = sample['x'], sample['y_exp']
                    output = net(data)
                    max_error = max(max_error, abs(output - label))
                    rms_error += (output - label) * (output - label)
                    x.append(data.item())
                    y.append(output.item())
                    y_exp.append(label.item())
            rms_error = math.sqrt(rms_error / eval_ds.size)
            eval_metric = -math.log10(rms_error)
            print("epoch ", str(epoch), " | eval metric: ", str(eval_metric), " | max error: ", str(max_error))

        # do training
        net.train()
        for i_batch, batch in enumerate(train_dataloader):
            optimizer.zero_grad()  # zero the gradient buffers
            data, label = batch['x'], batch['y_exp']
            output = net(data)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

    plt.scatter(x, y, s=1, c='red') # results
    plt.plot(x, y_exp, color='blue') # target function
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    set_random_seed.set_random_seed()
    parser = argparse.ArgumentParser(description='Script to train a function approximator')
    parser.add_argument('func_name', type=str, help='function name')
    args = parser.parse_args()
    main(args.func_name)
