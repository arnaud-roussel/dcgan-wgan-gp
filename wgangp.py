import torch
import torch.nn as nn

# From https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
def calc_gradient_penalty(D, real_data, fake_data):

    N = real_data.shape[0]

    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda:
        device = real_data.get_device()
    else:
        device = 'cpu'

    alpha = torch.rand(N, 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = D(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class D_loss(nn.Module):

    def __init__(self, lambd=10):
        super().__init__()
        self.lambd = lambd

    def forward(self, DGz, Dx, gradient_penalty):
        return DGz.mean() - Dx.mean() + self.lambd * gradient_penalty


class G_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, DGz):
        return -DGz.mean()