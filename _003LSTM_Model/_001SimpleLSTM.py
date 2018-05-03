import torch
from torch.autograd import Variable

# super params
input_size = 2
hidden_size = 5
time_steps = 3
batch_size = 2
num_layers = 2

# training/target data
inputs = Variable(torch.randn(time_steps, batch_size, input_size))
target = Variable(torch.LongTensor(batch_size).random_(hidden_size-1))

# the model
model = torch.nn.LSTM(input_size, hidden_size, num_layers)

# forward
outputs, _ = model(inputs)
last_output = outputs[-1]

# backward
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

loss = criterion(last_output, target)
#optimizer.zero_grad()
loss.backward()
#optimizer.step()

'''
print(model)
params = model.state_dict()
for k, v in params.items():
    print(k, v)
'''
