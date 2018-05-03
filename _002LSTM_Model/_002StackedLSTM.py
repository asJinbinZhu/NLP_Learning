import torch
from torch.autograd import Variable

torch.manual_seed(1)

# super params
input_size = 2
hidden_size = 5
time_steps = 2
batch_size = 2
num_layers = 2

# training/target data
inputs = Variable(torch.randn(time_steps, batch_size, input_size))
target = Variable(torch.LongTensor(batch_size).random_(0, hidden_size - 1))

def init_hidden():
    h = Variable(torch.randn(1, batch_size, hidden_size))
    c = Variable(torch.randn(1, batch_size, hidden_size))
    return h, c

# stacked lstm model
model = torch.nn.ModuleList()
for i in range(num_layers):
    input_size = input_size if i == 0 else hidden_size
    model.append(torch.nn.LSTM(input_size, hidden_size))
'''
print(model)
params = model.state_dict()
for k, v in params.items():
    print(k, v)
'''
# forward
outputs = []
for i in range(num_layers):
    outputs.clear()
    hidden = init_hidden()
    for input in inputs:
        input = input.unsqueeze(0)
        output, (h, c) = model[i](input, hidden)
        hidden = (h, c)
        outputs.append(output)

    inputs = (outputs[0].squeeze(0), outputs[1].squeeze(0))

last_output = output[-1]
#print('Last_output: ', last_output)

# backward
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

loss = criterion(last_output, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

'''
inputs:  Variable containing:
(0 ,.,.) = 
  0.6614  0.2669
  0.0617  0.6213

(1 ,.,.) = 
 -0.4519 -0.1661
 -1.5228  0.3817
[torch.FloatTensor of size 2x2x2]

Inner outputs:  [Variable containing:
(0 ,.,.) = 
 -0.0311  0.3875  0.4077 -0.3172 -0.1058
 -0.0224  0.4716 -0.0802 -0.2939 -0.0886
[torch.FloatTensor of size 1x2x5]
]
Inner outputs:  [Variable containing:
(0 ,.,.) = 
 -0.0311  0.3875  0.4077 -0.3172 -0.1058
 -0.0224  0.4716 -0.0802 -0.2939 -0.0886
[torch.FloatTensor of size 1x2x5]
, Variable containing:
(0 ,.,.) = 
 -0.1283 -0.0276  0.2129 -0.0769 -0.0400
 -0.1585 -0.1000 -0.1244  0.0242  0.0718
[torch.FloatTensor of size 1x2x5]
]
Inner outputs:  [Variable containing:
(0 ,.,.) = 
  0.0557  0.4276 -0.0920  0.1055 -0.0406
  0.4035 -0.2891 -0.0085  0.0421  0.1667
[torch.FloatTensor of size 1x2x5]
]
Inner outputs:  [Variable containing:
(0 ,.,.) = 
  0.0557  0.4276 -0.0920  0.1055 -0.0406
  0.4035 -0.2891 -0.0085  0.0421  0.1667
[torch.FloatTensor of size 1x2x5]
, Variable containing:
(0 ,.,.) = 
  0.0961  0.1655 -0.0583  0.0125  0.0463
  0.2701 -0.2240 -0.0100 -0.0379  0.1499
[torch.FloatTensor of size 1x2x5]
]

'''

