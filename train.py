import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from modules import Generator, Discriminator
from wgangp import D_loss, G_loss, calc_gradient_penalty
import sys

DOWNLOAD = False
BATCH_SIZE = 64
EPOCHS = 20
Z_DIM = 100
D_STEPS = 1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

t = transforms.Compose([transforms.Resize(64),
                        transforms.CenterCrop(64),
                        transforms.ToTensor(),
                        lambda x: x * 2 - 1])  # Scaling to -1, 1

dataset = torchvision.datasets.CelebA('G:/Datasets', download=DOWNLOAD, transform=t)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

G = Generator(Z_DIM).to(device)
D = Discriminator().to(device)
d_loss = D_loss().to(device)
g_loss = G_loss().to(device)

optim_D = torch.optim.Adam(D.parameters(), lr=1e-5, betas=(0.5, 0.999))
optim_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))

d_count = 0
for e in range(EPOCHS):
    for x, _ in dataloader:
        for p in D.parameters():
            p.requires_grad = True

        x = x.to(device)
        D.zero_grad()
        d_count += 1

        Dx = D(x)
        z = torch.randn((BATCH_SIZE, Z_DIM)).to(device)
        Gz = G(z)
        DGz = D(Gz)
        gradient_penalty = calc_gradient_penalty(D, x, Gz)
        d_cost = d_loss(DGz, Dx, gradient_penalty)
        d_cost.backward()
        optim_D.step()

        if (d_count % D_STEPS) == 0:
            for p in D.parameters():
                p.requires_grad = False
            G.zero_grad()
            z = torch.randn((BATCH_SIZE, Z_DIM)).to(device)
            Gz = G(z)
            DGz = D(Gz)
            g_cost = g_loss(DGz)
            g_cost.backward()
            optim_G.step()
            sys.stdout.write(f'\r {d_count} Epoch {e}/{EPOCHS} {d_count * 100 /len(dataloader):.2f}% '
                             f'D:{d_cost.item():.2f}, G:{g_cost.item():.2f} Dx: {Dx.mean().item():.2f} '
                             f'DGz: {DGz.mean().item():.2f}')

            if (d_count % 5000) == 0:
                Gz = (Gz + 1) / 2
                torchvision.utils.save_image(Gz, f'./sample_{d_count}.jpg')
    torch.save(G.state_dict(), 'G.nn')
    torch.save(D.state_dict(), 'D.nn')