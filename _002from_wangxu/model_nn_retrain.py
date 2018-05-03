#coding=utf-8
import torch
from torch import nn
from model_init_params import in_dim_layer,layer_1_init,layer_2_init,layer_3_init,prune_rate



which_neuron_hook_layer1=[]#定义全局变量，用于在hook调用中，指出删减的神经元的位置-layer1
which_neuron_hook_layer2=[]#定义全局变量，用于在hook调用中，指出删减的神经元的位置-layer2

#根据裁剪率，计算需要裁剪的神经元个数
total_neurons_number = layer_1_init + layer_2_init #MNIST数据集的输出神经元不进行裁剪
total_prune_neurons_number = int(prune_rate * total_neurons_number)#int为向下取整的内建函数


#根据不需要反向传播更新的神经元的数量，也就是需要裁剪的神经元数，从sorted_record.txt中自动调取相应的数量的神经元并近似
f = open('data_prune/sorted_record_small_to_big.txt','r')
while total_prune_neurons_number:
	line = f.readline()
	line_split_layer = int(filter(str.isdigit,line.split(',',2)[0]))#filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表
	
	line_split_neuron = int(filter(str.isdigit,line.split(',',2)[1]))
	if line_split_layer == 1:
		which_neuron_hook_layer1.append(line_split_neuron-1)#存放第一个隐藏层被裁减的神经元位置，注意神经line_split_neuron从1开始计数,所以需要减1
	elif line_split_layer == 2:
		which_neuron_hook_layer2.append(line_split_neuron-1)#存放第二个隐藏层被裁减的神经元位置
	
	total_prune_neurons_number -= 1
f.close()
def hook_hidden1_weight_grad(grad):#用于将weight的梯度置零,hidden1层需要将上下两层的权重梯度都置零，此hook用于将第一个隐藏层的上层连接的权重梯度置零
	global which_neuron_hook_layer1
	if which_neuron_hook_layer1:#用于判断证明列表是否为空，一个空列表等同于FALSE
		for i in which_neuron_hook_layer1:
			for j in range(len(grad[0])):
				grad.data[i][j]=0.0
		

def hook_hidden2_weight_grad(grad):
	global which_neuron_hook_layer1#用于将第一个隐藏层的下层连接的权重梯度置零
	if which_neuron_hook_layer1:#用于判断证明列表是否为空，一个空列表等同于FALSE
		for i in which_neuron_hook_layer1:
			for j in range(len(grad)):
				grad.data[j][i]=0.0
		
	global which_neuron_hook_layer2#用于将第二个隐藏层的上层连接的权重梯度置零
	if which_neuron_hook_layer2:#用于判断证明列表是否为空，一个空列表等同于FALSE
		for i in which_neuron_hook_layer2:
			for j in range(len(grad[0])):
				grad.data[i][j]=0.0				
def hook_hidden2_down_weight_grad(grad):#用于将第二个隐藏层的下一层（与输出层相连的）连接的权重梯度置零
	global which_neuron_hook_layer2#用于将第二个隐藏层的下层连接的权重梯度置零
	if which_neuron_hook_layer2:#用于判断证明列表是否为空，一个空列表等同于FALSE
		for i in which_neuron_hook_layer2:
			for j in range(len(grad)):
				grad.data[j][i]=0.0				





class MLP(nn.Module):
    def __init__(self, in_dim=in_dim_layer, n_hidden_1=layer_1_init, n_hidden_2=layer_2_init, out_dim=layer_3_init):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1,)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2,)
        self.layer3 = nn.Linear(n_hidden_2, out_dim,)

    def forward(self, x):
        x = self.layer1(x)
	#self.layer1.weight.register_hook(hook_hidden1_weight_grad)
	x = nn.functional.sigmoid(x)
	self.layer1.weight.register_hook(hook_hidden1_weight_grad)
        x = self.layer2(x)
	#self.layer2.weight.register_hook(hook_hidden2_weight_grad)
	x = nn.functional.sigmoid(x)
	self.layer2.weight.register_hook(hook_hidden2_weight_grad)
        x = self.layer3(x)
	#self.layer3.weight.register_hook(hook_hidden2_down_weight_grad)
	#x = nn.functional.sigmoid(x)
	self.layer3.weight.register_hook(hook_hidden2_down_weight_grad)
        return x	
