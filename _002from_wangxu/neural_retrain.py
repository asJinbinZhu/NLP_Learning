#coding=utf-8
import torch
from torch import nn, optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from model_nn_retrain import MLP 
from model_init_params import batch_size,class_num,learning_rate_retrain


train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

'''print('len train_dataset')
print(len(train_dataset))#60000
print('len test_dataset')
print(len(test_dataset))#10000
print('len train_loader')
print(len(train_loader))60000/batch_size
print('len test_loader')
print(len(test_loader))#10000/batch_size'''


def ConvertToOneHot(class_num,batch_size,label):
	#label = torch.LongTensor(batch_size,1).label % class_num
	label_onehot = torch.FloatTensor(batch_size,class_num)
	label_onehot.zero_()
	label_onehot.scatter_(1,label.data.unsqueeze(dim=1),1.0)
	return Variable(label_onehot)
	
#load the pruned model
model = MLP()
model.load_state_dict(torch.load('data_prune/mlp_pruned.pkl',map_location=lambda storage,loc:storage))

criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(model.parameters(), lr=learning_rate_retrain)

num_epoches=1
epoch_record=0
#for epoch in range(num_epoches):
while num_epoches:
    print('epoch {}'.format(epoch_record + 1))
    print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        img = img.view(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)
	label1 = label
        # 向前传播
        out = model(img)	
	#label = ConvertToOneHot(class_num,batch_size,label)
	# criterion.cuda()
        loss = criterion(out, label)
        running_loss += loss.data[0] * label1.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label1).sum()
        running_acc += num_correct.data[0]
        # 向后传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
	if i == 30:
		break
	
        if i % 10 == 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch_record + 1, 'n', running_loss / (batch_size * i),
                running_acc / (batch_size * i)))

 	if( running_acc/(batch_size*i)) >= 0.960383 or epoch_record>= 30:
		num_epoches=0
		print('epoch_record')
		print(epoch_record)
    print('uncuda Finish retrain {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
        epoch_record + 1, running_loss / (len(train_dataset)), running_acc / (len(train_dataset))))
    epoch_record += 1
  
#网络retrain一次后的精度
model.eval()
eval_loss = 0.
eval_acc = 0.
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)

    img = Variable(img, volatile=True)
    label = Variable(label, volatile=True)
    label1=label
    out = model(img)
    #label =ConvertToOneHot(class_num,1,label)
    #criterion.cuda()
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label1.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label1).sum()
    eval_acc += num_correct.data[0]
print('uncuda Test Loss of retrain: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc / (len(test_dataset))))
print()
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'*4)
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'*4)
print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'*4)

