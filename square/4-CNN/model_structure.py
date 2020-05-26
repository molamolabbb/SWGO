import torch
import torchviz
from torchviz import make_dot

def save_model_structure(model,modelnumber):
  x = torch.rand(1,4,51,51)
  y = model(x)

  dot = make_dot(y.mean(), params=dict(model.named_parameters()))
  dot.format='pdf'
  dot.render('save_model/best/structure/model'+modelnumber)
  #return dot

