import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, latent_dim = 100):

        super().__init__()
        self.lin_in = nn.Linear(latent_dim, 4 * 4 * 1024)
        self.main_sequence = nn.Sequential(nn.BatchNorm2d(1024),
                                  nn.ConvTranspose2d(1024, 512, 5, 2, 2, output_padding=1, bias=False),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(512),
                                  nn.ConvTranspose2d(512, 256, 5, 2, 2, output_padding=1, bias=False),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(256),
                                  nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1, bias=False),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(128, 3, 5, 2, 2, output_padding=1, bias=False),
                                  nn.Tanh())

    def forward(self, z):

        h = self.lin_in(z)
        h = torch.reshape(h, (-1, 1024, 4, 4))
        h = self.main_sequence(h)

        return h


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.main_sequence = nn.Sequential(nn.Conv2d(3, 128, 5, 2, 3),
                                  nn.LeakyReLU(0.2),
                                  nn.BatchNorm2d(128),
                                  nn.Conv2d(128, 256, 5, 2, 2),
                                  nn.LeakyReLU(0.2),
                                  nn.BatchNorm2d(256),
                                  nn.Conv2d(256, 512, 5, 2, 2),
                                  nn.LeakyReLU(0.2),
                                  nn.BatchNorm2d(512),
                                  nn.Conv2d(512, 1024, 5, 2, 1),
                                  nn.LeakyReLU(0.2),
                                  nn.BatchNorm2d(1024),
                                  nn.Conv2d(1024, 1, 4, 1, 0))

    def forward(self, x):

        h = self.main_sequence(x).reshape(-1, 1)
        return h