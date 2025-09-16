import numpy as np
import torch
import matplotlib.pyplot as plt
from setting import *
from numpy import linalg as LA
from torch import nn

def get_mgrid_2d(xlen, ylen):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    grid = torch.meshgrid(torch.linspace(-1, 1, steps=xlen), torch.linspace(-1, 1, steps=ylen))
    mgrid = torch.stack(grid, dim=-1)
    mgrid = mgrid.reshape(-1, 2)
    return mgrid

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def wave_eq(y, x):
    grad = gradient(y, x)
    grad[..., 0] = grad[..., 0]
    grad[..., 1] = grad[..., 1] * -1 * (1 / c ** 2) * ((sr / time_len) ** 2)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def plot_ir(ir, name="", type="_x"):
    xticks = np.array(range(ir.shape[1]))
    yticks = np.linspace(0, ir.shape[0]*1000/sr, ir.shape[0])
    plt.figure()
    plt.pcolor(xticks, yticks, ir)
    plt.colorbar()
    plt.clim(-0.3, 0.9)
    plt.gca().invert_yaxis()
    plt.ylabel("Time (ms)")
    plt.xlabel("Microphone")
    plt.savefig(name + ".png")

def plot_sf(ir, name):
    xticks = np.array(range(ir.shape[0]))
    yticks = np.array(range(ir.shape[1]))
    plt.figure()
    plt.pcolor(xticks, yticks, ir)
    plt.colorbar()
    plt.clim(-0.3, 0.9)
    plt.savefig(name + "_sf.png")

def time_loss(rir, truth):
    error = rir - truth
    squared_error = LA.norm(error, ord=2, axis=-1) ** 2
    denominator = LA.norm(truth, ord=2, axis=-1) ** 2
    normalised_error = squared_error / denominator
    mse = np.mean(normalised_error)
    return to_decibel(mse)

def to_decibel(value):
    return 10 * np.log10(value)

def frequency_visuliaze(num_mic, time_len, ir, name="", sr=44100, mic=5):
    fr = np.zeros((time_len//2+1, num_mic))
    for i in range(num_mic):
        fr[:, i] = np.abs(np.fft.rfft(ir[:, i]))
    fig, ax = plt.subplots()
    xticks = np.array(range(num_mic))
    yticks = np.linspace(0, sr, time_len//2+1)
    mesh = ax.pcolormesh(xticks, yticks, to_decibel(fr))
    mesh.set_clim(-50, 10)
    ax.set_yscale('log')
    fig.colorbar(mesh)
    ax.invert_yaxis()
    ax.set_ylabel("Frequency (Hz)")
    plt.savefig("./res/mic_" + str(mic) + "/" + str(sr) + "/" + name + "_f.png")


def frequency_loss(rir, truth):
    fr = np.zeros((rir.shape[0]//2+1, rir.shape[1]))
    for i in range(rir.shape[1]):
        fr[:, i] = np.abs(np.fft.rfft(rir[:, i])) ** 2
    fr_truth = np.zeros((truth.shape[0]//2+1, rir.shape[1]))
    for i in range(truth.shape[1]):
        fr_truth[:, i] = np.abs(np.fft.rfft(truth[:, i])) ** 2
    SD = to_decibel(fr / fr_truth)
    loss = np.abs(SD).mean(axis=1)
    omega = np.array([1/n for n in range(1, loss.shape[0]+1)])
    return np.sum(loss * omega) / np.sum(omega)


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(
            self,
            d_input: int,
            n_freqs: int,
            log_space: bool = False
    ):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * np.pi * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * np.pi * freq))

    def forward(
            self,
            x
    ) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)

