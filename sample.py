import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
from modules import Generator, Discriminator
from wgangp import D_loss, G_loss, calc_gradient_penalty
import sys

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
G = Generator().to(device)
G.load_state_dict(torch.load('G.nn'))

with torch.no_grad():
    z = torch.randn((64, 100)).to(device)
    Gz = G(z)
    Gz = (Gz + 1) / 2
    torchvision.utils.save_image(Gz, f'./sample.jpg')