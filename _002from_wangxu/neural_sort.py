#coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from model_nn import MLP
from model_init_params import layer_1_init,layer_2_init,layer_3_init,batch_size,learning_rate,class_num


dict_all_output_weight={}#创建字典，用于存放一张图片的所有的输出权重值
grad_output_list=[]#用于存放调用hook函数后，得到的grad_output值
all_dict_all_output_weight=[]#创建列表，用于存放一个batch_size个大小的图片的所有的输出权重值，且每组数据为一个大的字典元素。
each_position_all_dict_all_output_weight=[]#创建列表，用于存放从all_dict_all_output_weight获得的，每个相同位置的值存为一个列表
position_excluded=[]#总共大小为total_neurons_number，用于存储已经确定好排序位置的神经元的位置。此列表相当于用于存储基于统计排序方法的，最终的神经元重要程度排序的位置


def dictToSaveEachLayerOutputWeight():
	#number_neurons用于指示本层神经元的数量，与layer_x_init相对应
	which_layer=[]#当创建字典时，用于保存是哪层神经元
	which_neuron=[]#当创建字典时，用于保存是同层第几个神经元，从数字1开始
	which_layer_neuron=[]#当创建字典时，用于将层数和神经元位置数做组合
	dict_output_weight={}#创建字典，用于保存本层的字典的最终形式
	global grad_output_list
	for i in range(layer_2_init):#由于hook函数是从后层到前层的顺序输出的值，所以先创建后层的值
		which_layer.append(2)#创建与output weight为相同数量的列表
		which_neuron.append(i+1)#用于指出神经元的位置数
	for i in range(layer_1_init):
		which_layer.append(1)#创建与output weight为相同数量的列表
		which_neuron.append(i+1)#用于指出神经元的位置数
	#for i in range(layer_3_init):
		#which_layer.append(3)#创建与output weight为相同数量的列表
		#which_neuron.append(i+1)#用于指出神经元的位置数
	which_layer_neuron=list(zip(which_layer,which_neuron))
	dict_output_weight=dict(zip(which_layer_neuron,grad_output_list))
	global dict_all_output_weight
	dict_all_output_weight.update(dict_output_weight)#将一张图片所有的各层最终字典内容进行汇总
	global all_dict_all_output_weight
	all_dict_all_output_weight.append(dict_all_output_weight)#将一batch_size图片所有的各层最终字典内容进行汇总
	dict_all_output_weight={}#将字典清空，以存放下一张图片的内容。
	
	
	
def ConvertToOneHot(class_num,batch_size,label):
	#label = torch.LongTensor(batch_size,1).label % class_num
	label_onehot = torch.FloatTensor(batch_size,class_num)
	label_onehot.zero_()
	label_onehot.scatter_(1,label.data.unsqueeze(dim=1),1.0)
	return Variable(label_onehot)

def hook(module,grad_input,grad_output):
	for i,data in enumerate(grad_output,0):
		if i==0:#此语句无用
			grad_output=data
			global grad_output_list
			for j in range(len(grad_output[0])):
				#grad_output_list.extend(list(grad_output[0][j].data.abs().numpy()))
				grad_output_list.extend(list(grad_output[0][j].data.abs()))
		else:
			pass
	dictToSaveEachLayerOutputWeight()
	grad_output_list=[]#将此清零，用于保存下一张图片的数据
	
def hook1(module,grad_input,grad_output):
	for i,data in enumerate(grad_output,0):
		if i==0:
			grad_output=data
			global grad_output_list
			for j in range(len(grad_output[0])):
				#grad_output_list.extend(list(grad_output[0][j].data.abs().numpy()))
				grad_output_list.extend(list(grad_output[0][j].data.abs()))
		else:
			pass





#load the pretrained model
model = MLP()
model.load_state_dict(torch.load('data_prune/mlp_pretrain.pkl'))
if torch.cuda.is_available():
	model = model.cuda()
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
test_dataset = datasets.MNIST(root='./data',train=False,transform=transforms.ToTensor())	
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


for epoch in range(1):
	for i, data in enumerate(test_loader,0):
		if 0 <= i <(batch_size+2):#batch_size+2意味着共获得batch_size个图片的hook调用
			img,label= data
			img = img.view(img.size(0), -1)
			if torch.cuda.is_available():
				img=Variable(img).cuda()
				label=Variable(label).cuda()
			else:
				img=Variable(img)
				label=Variable(label)
			output = model(img)
			#label =ConvertToOneHot(class_num,1,label)
			criterion.cuda()
			loss = criterion(output,label)	
			optimizer.zero_grad()
			if i == 0:#i==0 时，意味着从第一张图片开始获取output_weight值.即选择从第几张图片开始进行
				handle = model.layer1.register_backward_hook(hook)
				handle = model.layer2.register_backward_hook(hook1)
				#handle = model.layer3.register_backward_hook(hook)
			loss.backward()
			#optimizer.step()只进行排序操作，不对预训练的参数进行修改
		else:
			pass
			
#将一batch_size图片所有的各层最终字典内容进行汇总后，将里面的内容进行排序
dict_sorted_all_output_weight=all_dict_all_output_weight#用于存放排序后的结果。现在赋值是为了确定其大小。
for i in range(batch_size):
	dict_sorted_all_output_weight[i] = sorted(all_dict_all_output_weight[i].items(),key=lambda item:item[1])


#将排序后的一个batch_size个大小的数据进行统计排序，并最终确立一个sorted顺序	
total_neurons_number = layer_1_init + layer_2_init
for j in range(total_neurons_number):#从total_neurons_number选择比对的神经元的位置
	for i in range(batch_size):#在确定好位置的情况下，选择batch_size个相同位置的参数
		each_position_all_dict_all_output_weight.append(dict_sorted_all_output_weight[i][j][0])
	#print('each_position_all_dict_all_output_weight :')
	#print(each_position_all_dict_all_output_weight)
    
	max_item=()
	max_count_item=0
	myset = set(each_position_all_dict_all_output_weight)#myset是另外一个列表，里面的内容是each_possition_all_dict_all_output_weight里面的无重复项
	for item in myset:
		if item not in position_excluded :
			if each_position_all_dict_all_output_weight.count(item) > max_count_item :
				max_item= item
				max_count_item=each_position_all_dict_all_output_weight.count(max_item)
	if (max_item != ())and (max_count_item != 0):#为零时说明该组位置数据都在position_excluded出现过，所以此时不能将空元组输入
		#print('the MAX ',max_item,'has found', max_count_item)
		position_excluded.append(max_item) #存入已经排好序的神经元位置
	for item in myset:#挑选出最大值后，再遍历一遍，防止有相同大小的值存在
		if (max_count_item != 0)and (max_item != ())  and (each_position_all_dict_all_output_weight.count(item) == max_count_item) and (item not in position_excluded) :
			#print('the MAX also has',item,'has found', each_position_all_dict_all_output_weight.count(item))
			position_excluded.append(item)#存入已经排好序的神经元位置
	if len(position_excluded) == total_neurons_number:#当值为400时说明已经排序完成，不需要再循环了
		break
	each_position_all_dict_all_output_weight=[]#置零，使其为空存储下一位置的参数

#将sort后的参数存储到文件中,small to big 
f = open('data_prune/sorted_record_small_to_big.txt','w')
for i  in position_excluded:
	f.write(str(i)+'\n')
f.close()

position_excluded=list(reversed(position_excluded))
#将sort后的参数存储到文件中,big to small
f = open('data_prune/sorted_record_big_to_small.txt','w')
for i  in position_excluded:
	f.write(str(i)+'\n')
f.close()

