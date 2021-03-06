import torch
import torchviz
from torchviz import make_dot
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def save_model_structure(model,modelnumber):
  device = torch.device('cuda')
  x = torch.rand(1,4,51,51)
  x = x.to(device)
  y = model(x)

  dot = make_dot(y.mean(), params=dict(model.named_parameters()))
  dot.format='pdf'
  dot.render('save_model/best/structure/model'+modelnumber)
  #return dot

