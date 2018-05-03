import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision.datasets as DataSet
import torchvision.transforms as Trans

# super params
input_size = 28
hidden_size = 64
time_steps = 28
batch_size = 64
num_layers = 1
num_classes = 10
num_epoch = 1

# training data
train_data = DataSet.MNIST(
    root = './mnist',
    train = True,
    transform = Trans.ToTensor(),
    download = True
)
train_loader = Data.DataLoader(dataset=train_data, batch_size = batch_size, shuffle=True)

# test data
test_data = DataSet.MNIST(root='./mnist', train=False, transform=Trans.ToTensor())
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:1]/255
test_y = test_data.test_labels.numpy().squeeze()[:1]

# the model
class Simple_LSTM_Classifier_for_MNIST(torch.nn.Module):
    def __init__(self):
        super(Simple_LSTM_Classifier_for_MNIST, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(hidden_size, num_classes)

    def init_hidden(self):
        return (
            Variable(torch.randn(num_layers, batch_size, hidden_size)),
            Variable(torch.randn(num_layers, batch_size, hidden_size))
        )

    def forward(self, x):
        hidden = self.init_hidden()
        output_lstm, (h, c) = self.lstm(x, hidden)
        output_linear = self.classifier(output_lstm[:, -1, :])
        return output_linear

model = Simple_LSTM_Classifier_for_MNIST()
'''
params = model.state_dict()
for k, v in params.items():
    print(k, v)
'''

# training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

for epoch in range(num_epoch):
    # forward
    for step, (x, y) in enumerate(train_loader):
        print('step: ', step, 'x: ', x, 'y: ', y)
