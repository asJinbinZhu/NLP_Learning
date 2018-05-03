#coding=utf-8
import torch
from torch import nn
from model_init_params import in_dim_layer,layer_1_init,layer_2_init,layer_3_init


class MLP(nn.Module):
    def __init__(self, in_dim=in_dim_layer, n_hidden_1=layer_1_init, n_hidden_2=layer_2_init, out_dim=layer_3_init):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1,)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2,)
        self.layer3 = nn.Linear(n_hidden_2, out_dim,)

    def forward(self, x):
        x = self.layer1(x)
	x = nn.functional.sigmoid(x)
        x = self.layer2(x)
	x = nn.functional.sigmoid(x)
        x = self.layer3(x)
	#x = nn.functional.sigmoid(x)
        return x	
