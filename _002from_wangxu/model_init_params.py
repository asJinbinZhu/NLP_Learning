#coding=utf-8

#初始化各层神经元数量， 包括model.nn和model.nn.retrain
in_dim_layer=28*28
layer_1_init=300
layer_2_init=100
layer_3_init=10

#设置裁剪率
prune_rate=0.15

#设置网络超参，此为pretrain的参数，neural_sort的参数，neural_pruning的参数，和neural_retrain的部分参数
batch_size =32	
learning_rate = 1e-2#pretrain的学习率
num_epoches = 100#pretrain的epoch次数
class_num = 10 #指出神经网络的分类数量。mnist数据集的输出是0-9

#设置retrain的learning rate
learning_rate_retrain=1e-3#小于learning rate十倍以上
