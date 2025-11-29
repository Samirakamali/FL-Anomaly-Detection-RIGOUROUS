import torch
from torch import nn
import torch.nn.functional as F

#-------------------------------- Model for WUSTL-IIoT dataset --------------------------------------------
# class MLP(nn.Module): 
#   def __init__(self, args):
#     super(MLP,self).__init__()
#     self.fc1 = nn.Linear(in_features=41 , out_features=9 , bias=True)
#     #self.dropout = nn.Dropout(0.3)
#     self.fc2 = nn.Linear(in_features=9 , out_features=9 , bias=True)
#     self.fc3 = nn.Linear(in_features=9 , out_features=args.num_classes , bias=True) #multiclass  out_features=5

#   def forward(self,x):
#     out = self.fc1(x)
#     out = F.relu(out)
#     #out = self.dropout(out)
#     out = self.fc2(out)
#     out = F.relu(out)
#     out = self.fc3(out)
#     return out

#-------------------------------- Model for Sandbox dataset --------------------------------------------
class MLP(nn.Module): 
  def __init__(self, args):
    super(MLP,self).__init__()
    self.fc1 = nn.Linear(in_features=29 , out_features=64 , bias=True)
    #self.dropout = nn.Dropout(0.3)
    self.fc2 = nn.Linear(in_features=64 , out_features=32 , bias=True)
    self.fc3 = nn.Linear(in_features=32 , out_features= 6 , bias=True) #multiclass  out_features=6

  def forward(self,x):
    out = self.fc1(x)
    out = F.relu(out)
    #out = self.dropout(out)
    out = self.fc2(out)
    out = F.relu(out)
    out = self.fc3(out)
    return out
