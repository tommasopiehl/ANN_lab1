import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#data params
t0 = 301
tn = 1500
beta = 0.2
lam = 0.1
n = 10
delay = 5
batch_size = tn-t0+1
noise = False

def generate_data(t0, tn, beta, lam, n, delay, noise=noise):

    if noise == False:

        x = np.zeros(tn+delay)
        x_split = np.zeros([tn, delay-1])
        labels = []
        x[0] = 1.5

        for t in range(1, tn):
            x[t] = x[t-1] + (beta*x[t-delay])/(1+x[t-delay]**n) - lam*x[t-1]

        for t in range(t0-delay*(delay-1), tn):
            points = []
            for j in range(4):
                points.append(x[t-delay*j])
            x_split[t] = np.array(points)

        x_split = x_split[t0:]

        for t in range(t0, len(x)-delay):
            labels.append(x[t+delay])

        labels = np.array(labels)
        labels.reshape(-1, 1)

    return x_split, labels

def plot_data(data, t0, tn):

    x_range = np.linspace(t0,tn,num=tn-t0)
    plt.plot(x_range, data)
    plt.show()

#Split data
data, labels = generate_data(t0, tn, beta, lam, n, delay)

#for plotting data (task 4.3.1)
#plot_data(labels, t0, tn)

#800 samples in train-set
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=(400/len(data)), random_state=42)

#200 samples for validation and test respectively
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#set up the data sets
X_train= torch.tensor(X_train, dtype=torch.float32) 
y_train = torch.tensor(y_train, dtype=torch.float32)
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

X_test= torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

#The Network
#model params
input_size = 4
hidden_layers = 2
hidden_sizes = [2, 4] 
hidden_sizes = np.array(hidden_sizes)
output_size = 1
p_drop = 0.05
random_init = False

class MackeyGlassModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, p_drop, random_init):
        super(MackeyGlassModel, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.droup_out = nn.Dropout(p_drop)

        #nn.Linear automatically handles bias term
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if random_init == False:
            nn.init.xavier_normal_(self.hidden_layers[0].weight)
        if random_init == True:
            nn.init.normal_(self.hidden_layers[0].weight, mean=0, std=0.5)
            nn.init.normal_(self.hidden_layers[0].bias, mean=0, std=0.5)

        if hidden_layers == 1:
            self.hidden_layers.append(nn.Sigmoid())
        else:
            for i in range(1, hidden_layers):
                #nn.Linear automatically handles bias term
                self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                if random_init == False:
                    nn.init.xavier_normal_(self.hidden_layers[-1].weight)
                if random_init == True:
                    nn.init.normal_(self.hidden_layers[-1].weight, mean=0, std=0.5)
                    nn.init.normal_(self.hidden_layers[-1].bias, mean=0, std=0.5)
                self.hidden_layers.append(nn.Sigmoid())

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        if random_init == False:
                nn.init.xavier_normal_(self.output_layer.weight)
        if random_init == True:
                nn.init.normal_(self.output_layer.weight, mean=0, std=0.5)
                nn.init.normal_(self.output_layer.bias, mean=0, std=0.5)

    def forward(self, x):

        count = 0
        #Iteratively pass through each layer
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.droup_out(x)
            count += 1

        output_final = self.output_layer(x)
        return output_final

criterion = nn.MSELoss()

def multi_training(model_a, model_b, model_c, model_d, optimizer_a, optimizer_b, optimizer_c, optimizer_d, noise, criterion=criterion):

    conv_epochs = np.ones(4)*-1
    conv_MSE = np.zeros(4)

    loss_ls_a = []
    loss_ls_b = []
    loss_ls_c = []
    loss_ls_d = []

    train_a = []
    train_b = []
    train_c = []
    train_d = []

    for epoch in range(num_epochs):

        for batch_inputs, batch_labels in train_loader:
            inputs = batch_inputs.clone().detach()
            if noise != 0:
                noise_term = torch.randn_like(inputs) * noise
                inputs += noise_term
            labels = batch_labels.clone().detach().view(len(inputs), 1)
            optimizer_a.zero_grad()
            optimizer_b.zero_grad()
            optimizer_c.zero_grad()
            optimizer_d.zero_grad()

            #perform on train set

            outputs_a = model_a(inputs)
            outputs_b = model_b(inputs)
            outputs_c = model_c(inputs)
            outputs_d = model_d(inputs)

            loss_a = criterion(outputs_a, labels)
            loss_b = criterion(outputs_b, labels)
            loss_c = criterion(outputs_c, labels)
            loss_d = criterion(outputs_d, labels)

            train_a.append(loss_a.detach().numpy())
            train_b.append(loss_b.detach().numpy())
            train_c.append(loss_c.detach().numpy())
            train_d.append(loss_d.detach().numpy())

            loss_a.backward()
            loss_b.backward()
            loss_c.backward()
            loss_d.backward()

            optimizer_a.step()
            optimizer_b.step()
            optimizer_c.step()
            optimizer_d.step()

        for batch_inputs, batch_labels in val_loader:

            val_inputs = batch_inputs.clone().detach()
            val_labels = batch_labels.clone().detach().view(len(val_inputs), 1)

            #perform on val set

            val_outputs_a = model_a(val_inputs)
            val_outputs_b = model_b(val_inputs)
            val_outputs_c = model_c(val_inputs)
            val_outputs_d = model_d(val_inputs)

            loss_val_a = criterion(val_outputs_a, val_labels)
            loss_val_b = criterion(val_outputs_b, val_labels)
            loss_val_c = criterion(val_outputs_c, val_labels)
            loss_val_d = criterion(val_outputs_d, val_labels)

            loss_ls_a.append(loss_val_a.detach().numpy())
            loss_ls_b.append(loss_val_b.detach().numpy())
            loss_ls_c.append(loss_val_c.detach().numpy())
            loss_ls_d.append(loss_val_d.detach().numpy())

    print('Training complete!')

    return loss_ls_a, loss_ls_b, loss_ls_c, loss_ls_d, train_a, train_b, train_c, train_d, conv_epochs, conv_MSE

n_runs = 40

#Gaussian noise in train data (task 4.3.4)
noise = 0.0

num_epochs = 1200
criterion = nn.MSELoss()

#model params
input_size = 4
hidden_layers = 2
output_size = 1
p_drop = 0.05
random_init = False

#for comparison of amount of hidden nodes (task 4.3.2)
#Balanced/low
hidden_sizes_a = [3, 3] 
hidden_sizes_a = np.array(hidden_sizes_a)

#Balanced/high
hidden_sizes_b = [6, 6] 
hidden_sizes_b = np.array(hidden_sizes_b)

#Increase
hidden_sizes_c = [3, 6] 
hidden_sizes_c = np.array(hidden_sizes_c)

#Decrease
hidden_sizes_d = [6, 3] 
hidden_sizes_d = np.array(hidden_sizes_d)

w_decay = 1e-4
lr = 0.001

avg_a = []
avg_b = []
avg_c = []
avg_d = []

avg_train_a = []
avg_train_b = []
avg_train_c = []
avg_train_d = []

avg_conv = np.zeros(4)
avg_mse_conv = np.zeros(4)

run_count = 0

for i in range(n_runs):
    seed = i
    torch.manual_seed(seed)
    np.random.seed(seed)

    #Multiple models
    model_a = MackeyGlassModel(input_size, hidden_sizes_a, output_size, p_drop, False)
    model_b = MackeyGlassModel(input_size, hidden_sizes_b, output_size, p_drop, False)
    model_c = MackeyGlassModel(input_size, hidden_sizes_c, output_size, p_drop, False)
    model_d = MackeyGlassModel(input_size, hidden_sizes_d, output_size, p_drop, False)

    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=lr, weight_decay=w_decay)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=lr, weight_decay=w_decay)
    optimizer_c = torch.optim.Adam(model_c.parameters(), lr=lr, weight_decay=w_decay)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=lr, weight_decay=w_decay)

    loss_ls_a, loss_ls_b, loss_ls_c, loss_ls_d, train_a, train_b, train_c, train_d, conv_ls, MSE_ls = multi_training(model_a, model_b, model_c, model_d, optimizer_a, optimizer_b, optimizer_c, optimizer_d, noise, criterion=criterion)

    avg_a.append(loss_ls_a)
    avg_b.append(loss_ls_b)
    avg_c.append(loss_ls_c)
    avg_d.append(loss_ls_d)

    avg_train_a.append(train_a)
    avg_train_b.append(train_b)
    avg_train_c.append(train_c)
    avg_train_d.append(train_d)

    for i in range(len(conv_ls)):
        avg_conv[i] += conv_ls[i]/n_runs
        avg_mse_conv[i] += MSE_ls[i]/n_runs

    run_count += 1
    print(run_count)

x_range = np.linspace(0, len(avg_a[0]), num=len(avg_a[0]))

avg_a_all = [sum(x)/n_runs for x in zip(*avg_a)]
avg_b_all = [sum(x)/n_runs for x in zip(*avg_b)]
avg_c_all = [sum(x)/n_runs for x in zip(*avg_c)]
avg_d_all = [sum(x)/n_runs for x in zip(*avg_d)]

train_a_all = [sum(x)/n_runs for x in zip(*avg_train_a)]
train_b_all = [sum(x)/n_runs for x in zip(*avg_train_b)]
train_c_all = [sum(x)/n_runs for x in zip(*avg_train_c)]
train_d_all = [sum(x)/n_runs for x in zip(*avg_train_d)]

print("avg A", avg_a_all[-1])
print("avg B", avg_b_all[-1])
print("avg C", avg_c_all[-1])
print("avg D", avg_d_all[-1])

plt.plot(x_range, avg_a_all, c='b') #[6,6] or w = 0
plt.plot(x_range, avg_b_all, c='r') #[6,3] or w = 1e-4
plt.plot(x_range, avg_c_all, c='g') #[6,5] or w = 0.5
plt.plot(x_range, avg_d_all, c='c') #[6,9]
plt.title("Avg validation error for varying amount of units in second layer")
plt.show()

#Testing
# n_runs = 20
# test_a = []
# test_b = []

# loss_a = 0
# loss_b = 0

# model_a = MackeyGlassModel(input_size, hidden_sizes_a, output_size, p_drop, False)
# model_b = MackeyGlassModel(input_size, hidden_sizes_b, output_size, p_drop, False)
# model_c = MackeyGlassModel(input_size, hidden_sizes_a, output_size, p_drop, False)
# model_d = MackeyGlassModel(input_size, hidden_sizes_b, output_size, p_drop, False)

# optimizer_a = torch.optim.Adam(model_a.parameters(), lr=lr, weight_decay=w_decay)
# optimizer_b = torch.optim.Adam(model_b.parameters(), lr=lr, weight_decay=w_decay)
# optimizer_c = torch.optim.Adam(model_c.parameters(), lr=lr, weight_decay=w_decay)
# optimizer_d = torch.optim.Adam(model_d.parameters(), lr=lr, weight_decay=w_decay)


# for i in range(n_runs):

#     seed = i
#     torch.manual_seed(seed)
#     np.random.seed(seed)

#     loss_ls_a, loss_ls_b, loss_ls_c, loss_ls_d, train_a, train_b, train_c, train_d, conv_ls, MSE_ls = multi_training(model_a, model_b, model_c, model_d, optimizer_a, optimizer_b, optimizer_c, optimizer_d, criterion=criterion)
#     with torch.no_grad():
#         test_inputs = X_test.clone().detach()
#         predicted_outputs_a = model_a(test_inputs)
#         predicted_outputs_b = model_b(test_inputs)

#     test_labels = y_test.clone().detach().view(len(test_inputs), 1)
#     test_loss_a = criterion(predicted_outputs_a, test_labels)
#     test_loss_b = criterion(predicted_outputs_b, test_labels)

#     loss_a += test_loss_a.detach().numpy()/n_runs
#     loss_b += test_loss_b.detach().numpy()/n_runs

#     pred_a = predicted_outputs_a.detach().numpy()
#     pred_b = predicted_outputs_b.detach().numpy()
#     test_a.append(pred_a)
#     test_b.append(pred_b)

# avg_a_all = [sum(x)/n_runs for x in zip(*test_a)]
# avg_b_all = [sum(x)/n_runs for x in zip(*test_b)]

# print("A:", loss_a)
# print("B:", loss_b)

# x_range = np.linspace(0, len(test_labels), num=len(test_labels))
# plt.plot(x_range, test_labels, c='g')
# plt.plot(x_range, avg_a_all, c='b')
# plt.show()
# plt.plot(x_range, test_labels, c='g')
# plt.plot(x_range, avg_b_all, c='r')
# plt.show()






    
