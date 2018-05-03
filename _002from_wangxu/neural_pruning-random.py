#coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from torch import nn, optim
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from model_nn import MLP 
from model_init_params import layer_1_init,layer_2_init,batch_size,learning_rate,num_epoches,class_num,prune_rate
import random
import linecache

# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


def ConvertToOneHot(class_num,batch_size,label):
	#label = torch.LongTensor(batch_size,1).label % class_num
	label_onehot = torch.FloatTensor(batch_size,class_num)
	label_onehot.zero_()
	label_onehot.scatter_(1,label.data.unsqueeze(dim=1),1.0)
	return Variable(label_onehot)

	
def weightPrune(which_layer,which_neuron):
	state_dict = model.state_dict()
	if which_layer == 1:
		layer_weight_name_upper = 'layer1.weight'
		layer_weight_name_lower = 'layer2.weight'
	if which_layer == 2:
		layer_weight_name_upper = 'layer2.weight'
		layer_weight_name_lower = 'layer3.weight'
		
	layer_weight_upper = list(state_dict[layer_weight_name_upper])
	for i in range(len(layer_weight_upper)):
		for j in range(len(layer_weight_upper[i])):
			if i == which_neuron-1:#减一是因为神经元which_neuron范围从1-10，但是此时排序是从0-9
				layer_weight_upper[i][j] = 0
	layer_weight_upper = tuple(layer_weight_upper)
	dict_layer_weight_upper = {layer_weight_name_upper:layer_weight_upper}
	state_dict.update(dict_layer_weight_upper)
	
	layer_weight_lower = list(state_dict[layer_weight_name_lower])
	for i in range(len(layer_weight_lower)):
		for j in range(len(layer_weight_lower[i])):
			if j == which_neuron-1:
				layer_weight_lower[i][j] = 0
	layer_weight_lower = tuple(layer_weight_lower)
	dict_layer_weight_lower = {layer_weight_name_lower:layer_weight_lower}
	state_dict.update(dict_layer_weight_lower)
	

#load the pretrained model
model = MLP()
model.load_state_dict(torch.load('data_prune/mlp_pretrain.pkl'))
if torch.cuda.is_available():
	model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#网络不进行裁剪的精度
model.eval()
eval_loss = 0.
eval_acc = 0.
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
    else:
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
    label1=label
    out = model(img)
    #label =ConvertToOneHot(class_num,1,label)
    criterion.cuda()
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label1.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label1).sum()
    eval_acc += num_correct.data[0]
print('random Test Loss of unprune: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc / (len(test_dataset))))
print()

#根据裁剪率，计算需要裁剪的神经元个数
total_neurons_number = layer_1_init + layer_2_init #MNIST数据集的输出神经元不进行裁剪
total_prune_neurons_number = int(prune_rate * total_neurons_number)#int为向下取整的内建函数


#根据需要裁剪的神经元数，从sorted_record.txt中自动调取相应的数量的神经元并近似
#f = open('data_prune/unsorted_record_one_picture.txt','r')

total_neurons_number_sample=range(total_neurons_number-1)#设定需要随机抽取数的范围
sample_list=random.sample(total_neurons_number_sample,total_prune_neurons_number)#随机选择total_prune_neurons_number个参数并存入列表
for i in sample_list:
	line = linecache.getline('data_prune/unsorted_record_one_picture.txt',i)
	#line = f.readline()
	line_split_layer = int(float(filter(str.isdigit,line.split(',',2)[0])))#filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
	
	line_split_neuron = int(float(filter(str.isdigit,line.split(',',2)[1])))
	weightPrune(line_split_layer,line_split_neuron)	

	
#网络裁剪后的精度
eval_loss = 0.
eval_acc = 0.
for data in test_loader:
    img, label = data
    img = img.view(img.size(0), -1)
    if torch.cuda.is_available():
        img = Variable(img, volatile=True).cuda()
        label = Variable(label, volatile=True).cuda()
    else:
        img = Variable(img, volatile=True)
        label = Variable(label, volatile=True)
    label1=label
    out = model(img)
    #label =ConvertToOneHot(class_num,1,label)
    criterion.cuda()
    loss = criterion(out, label)
    eval_loss += loss.data[0] * label1.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label1).sum()
    eval_acc += num_correct.data[0]
print('random Test Loss of prune: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    test_dataset)), eval_acc / (len(test_dataset))))
print()
print('##################################################################')
print('##################################################################')
print('##################################################################')





