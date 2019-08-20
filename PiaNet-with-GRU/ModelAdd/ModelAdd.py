#PiaNet add GRU
import torch
from torch import nn
from collections import OrderedDict

PiaNet_ckpt_dir = './model_1.pth'
PiaNet_GRU_model = './PiaNet_GRU.pth'
#load saved PiaNet model and initilized GRU by xavier_uniform_
PiaNet_dict = torch.load(PiaNet_ckpt_dir)
#Using xavier_uniform_ since GRU use Tanh as activiaion function
PiaNet_dict["decoder1.input.0.weight"] = nn.init.xavier_uniform_(torch.empty([2, 1, 1, 1, 1]))
PiaNet_dict["decoder1.input.0.bias"] = nn.init.constant_(torch.empty(2), 0)
PiaNet_dict["decoder1.input.3.weight"] = nn.init.xavier_uniform_(torch.empty([2, 2, 1, 1, 1]))
PiaNet_dict["decoder1.input.3.bias"] = nn.init.constant_(torch.empty(2), 0)
PiaNet_dict["decoder1.GRU.GRUCell.conv3d.weight"] = nn.init.xavier_uniform_(torch.empty([2, 2, 3, 3, 3]))
PiaNet_dict["decoder1.GRU.GRUCell.conv3d.bias"] = nn.init.constant_(torch.empty(2), 0)
PiaNet_dict["decoder2.input.0.weight"] = nn.init.xavier_uniform_(torch.empty([2, 1, 1, 1, 1]))
PiaNet_dict["decoder2.input.0.bias"] = nn.init.constant_(torch.empty(2), 0)
PiaNet_dict["decoder2.input.3.weight"] = nn.init.xavier_uniform_(torch.empty([2, 2, 1, 1, 1]))
PiaNet_dict["decoder2.input.3.bias"] = nn.init.constant_(torch.empty(2), 0)
PiaNet_dict["decoder2.GRU.GRUCell.conv3d.weight"] = nn.init.xavier_uniform_(torch.empty([2, 2, 3, 3, 3]))
PiaNet_dict["decoder2.GRU.GRUCell.conv3d.bias"] = nn.init.constant_(torch.empty(2), 0)

# Resaved the extended model
torch.save(PiaNet_dict, PiaNet_GRU_model)

# Reload and check the results
PiaNet_GRU = torch.load(PiaNet_GRU_model)
keys = OrderedDict.fromkeys(PiaNet_GRU)
for key in keys:
    print(key)
print(PiaNet_GRU["decoder2.GRU.GRUCell.conv3d.weight"])
print(PiaNet_GRU["decoder2.GRU.GRUCell.conv3d.bias"])


