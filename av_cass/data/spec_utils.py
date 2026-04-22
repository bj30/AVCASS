
import torch
import numpy as np


istft_kwargs = dict(
    n_fft=510, hop_length=256, window='hann',
)
stft_kwargs = istft_kwargs
spec_factor=0.15
spec_abs_exponent=0.5

def spec_fwd(audio):

    spec = torch.stft(audio, **stft_kwargs)
    if spec_abs_exponent != 1:
        e = spec_abs_exponent
        spec = spec.abs()**e * torch.exp(1j * spec.angle())
    return spec * spec_factor

def spec_fwd_numpy(spec):
    if spec_abs_exponent != 1:
        e = spec_abs_exponent
        spec = np.abs(spec)**e * np.exp(1j * np.angle(spec))
    return spec * spec_factor

def spec_back(spec):
    spec = spec / spec_factor
    if spec_abs_exponent != 1:
        e = spec_abs_exponent
        spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
    return spec

def spec_back_numpy(spec):
    spec = spec / spec_factor
    if spec_abs_exponent != 1:
        e = spec_abs_exponent
        spec = np.abs(spec)**(1/e) * np.exp(1j * np.angle(spec))
    return spec