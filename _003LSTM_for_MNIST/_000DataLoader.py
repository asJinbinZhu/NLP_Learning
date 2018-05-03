import torch
import torch.utils.data as Data

torch.manual_seed(1)

batch_size = 5

# prepare data
x = torch.linspace(1, 10, 10)
y = torch.linspace(1, 10, 10)

# 1 - convert data to tensor
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

# 2 - import into dataloader
data_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=1
)

for epoch in range(1):
    for step, (batch_x, batch_y) in enumerate(data_loader):
        print('Step: ', step, 'batch_x: ', batch_x, 'batch_y: ', batch_y)

'''
print('x: ', x)
print('y: ', y)
x:  
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
[torch.FloatTensor of size 10]

y:  
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
[torch.FloatTensor of size 10]

# shuffle = False
print('Step: ', step, 'batch_x: ', batch_x, 'batch_y: ', batch_y)
Step:  0 batch_x:  
 1
 2
 3
 4
 5
[torch.DoubleTensor of size 5]
 batch_y:  
 1
 2
 3
 4
 5
[torch.DoubleTensor of size 5]

Step:  1 batch_x:  
  6
  7
  8
  9
 10
[torch.DoubleTensor of size 5]
 batch_y:  
  6
  7
  8
  9
 10
[torch.DoubleTensor of size 5]

# shuffle = True
print('Step: ', step, 'batch_x: ', batch_x, 'batch_y: ', batch_y)
Step:  0 batch_x:  
  5
  7
 10
  3
  4
[torch.DoubleTensor of size 5]
 batch_y:  
  5
  7
 10
  3
  4
[torch.DoubleTensor of size 5]

Step:  1 batch_x:  
 2
 1
 8
 9
 6
[torch.DoubleTensor of size 5]
 batch_y:  
 2
 1
 8
 9
 6
[torch.DoubleTensor of size 5]
'''
