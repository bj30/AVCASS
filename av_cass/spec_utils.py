import torch
from einops import rearrange

tanh_scale_factor = 1.

def wav2spec(waveforms, normalized=True):
    """
    waveforms: [batch_size, N, 65536]
    return: [batch_size, N, 2, 256, 256]
    """
    assert waveforms.dim() == 3 
    B, N, L = waveforms.shape
    assert L == 65536 - 256
    device = waveforms.device

    wav_spec = torch.stft(waveforms.view(-1, waveforms.size(-1)), n_fft=510, hop_length=256, win_length=510, window=torch.hann_window(510).to(device), return_complex=True, center=True, normalized=normalized)
    wav_spec = torch.view_as_real(wav_spec)
    # wav_spec: [batch_size * 4, 257, 128, 2]
    wav_spec = rearrange(wav_spec, '(b n) h w c -> b n c h w', b=B, n=N)
    wav_spec = torch.tanh(wav_spec * tanh_scale_factor)
    return wav_spec

def spec2wav(wav_spec, normalized=True):
    """
    wav_spec: [batch_size, N, 2, 256, 256]
    return: [batch_size, N, 65536]
    """
    device = wav_spec.device

    # clip the wav_spec to [-0.995, 0.995]
    # wav_spec = torch.clamp(wav_spec, -0.999, 0.999)

    wav_spec = torch.atanh(wav_spec) / tanh_scale_factor
    B, C, H, W = wav_spec.shape
    N = C // 2
    wav_spec = rearrange(wav_spec, 'b (n c) h w -> (b n) h w c', b=B, n=N)
    wav_spec = torch.view_as_complex(wav_spec.contiguous())
    waveforms = torch.istft(wav_spec, n_fft=510, hop_length=256, win_length=510, window=torch.hann_window(510).to(device), center=True, normalized=normalized)
    waveforms = waveforms.view(B, N, -1)
    return waveforms


spec_factor=0.15
spec_abs_exponent=0.5

def wav2spec_sto(waveforms, normalized=False):
    """
    waveforms: [batch_size, N, 130816]
    return: [batch_size, N, 2, 256, 256]
    """
    assert waveforms.dim() == 3 
    B, N, L = waveforms.shape
    device = waveforms.device

    wav_spec = torch.stft(waveforms.view(-1, waveforms.size(-1)), n_fft=510, hop_length=256, win_length=510, window=torch.hann_window(510).to(device), return_complex=True, center=True, normalized=normalized)
    e = spec_abs_exponent
    wav_spec = wav_spec.abs()**e * torch.exp(1j * wav_spec.angle()) * spec_factor

    wav_spec = torch.view_as_real(wav_spec)
    # wav_spec: [batch_size * 4, 257, 128, 2]
    wav_spec = rearrange(wav_spec, '(b n) h w c -> b n c h w', b=B, n=N)
    return wav_spec

def spec2wav_sto(wav_spec, normalized=False):
    """
    return: [batch_size, N, 130816]
    """
    device = wav_spec.device

    B, C, H, W = wav_spec.shape
    N = C // 2
    wav_spec = rearrange(wav_spec, 'b (n c) h w -> (b n) h w c', b=B, n=N)
    wav_spec = torch.view_as_complex(wav_spec.contiguous())

    wav_spec = wav_spec / spec_factor
    e = spec_abs_exponent
    wav_spec = wav_spec.abs()**(1/e) * torch.exp(1j * wav_spec.angle())

    waveforms = torch.istft(wav_spec, n_fft=510, hop_length=256, win_length=510, window=torch.hann_window(510).to(device), center=True, normalized=normalized)
    waveforms = waveforms.view(B, N, -1)
    return waveforms



# min_amp = 1e-3
# def audio2spec(waveforms):
#     """
#     audio: [batch_size, N, T]
#     return: [batch_size, N, 2, 256, T']
#     """
#     assert waveforms.dim() == 3 
#     B, N, L = waveforms.shape
#     device = waveforms.device

#     # for wo normalized version
#     normalize_parameter = [5.262296676635742, 5.324204921722412, 5.306260585784912, 5.324747085571289]
#     # for normalized version
#     # normalize_parameter = [2.2909603118896484, 2.3514132499694824, 2.299173355102539, 2.3514344692230225]
#     normalize_parameter = torch.tensor(normalize_parameter[:N]).to(device).view(1, N, 1, 1)


#     wav_spec = torch.stft(
#         waveforms.view(-1, waveforms.size(-1)), 
#         n_fft=510, 
#         hop_length=256, 
#         win_length=510, 
#         window=torch.hann_window(510).to(device), 
#         return_complex=True, center=True, 
#         normalized=False)
#     _, H, W = wav_spec.shape
#     wav_spec = wav_spec.reshape(B, N, H, W)
    
#     amp = torch.log1p(wav_spec.abs() + min_amp)
#     amp_factor = torch.log1p(torch.tensor(min_amp))
#     amp = (amp - amp_factor) / (normalize_parameter - amp_factor)
#     amp = torch.clamp(amp, 0, 1)
#     amp = amp * 2. - 1.

#     phase = wav_spec.angle() / torch.pi

#     output = torch.stack([amp, phase], dim=2)
#     return output

# def spec2audio(spec):
#     """
#     spec: [batch_size, N, 2, 256, T']
#     return: [batch_size, N, T]
#     """
#     device = spec.device
#     if spec.dim() == 4:
#         spec = spec.reshape(spec.size(0), -1, 2, spec.size(-2), spec.size(-1))
#     B, N, _, _, _ = spec.shape

#     # for wo normalized version
#     normalize_parameter = [5.262296676635742, 5.324204921722412, 5.306260585784912, 5.324747085571289]
#     # for normalized version
#     # normalize_parameter = [2.2909603118896484, 2.3514132499694824, 2.299173355102539, 2.3514344692230225]
#     normalize_parameter = torch.tensor(normalize_parameter[:N]).to(device).view(1, N, 1, 1)
    
#     amp = spec[:, :, 0, :, :]
#     amp = torch.clamp(amp, -1, 1)
#     amp = (amp + 1.) / 2.
#     amp_factor = torch.log1p(torch.tensor(min_amp))
#     amp = amp * (normalize_parameter - amp_factor) + amp_factor
#     amp = torch.expm1(amp) - min_amp
#     # make sure its all larger than 0
#     amp = torch.clamp(amp, 0, 1e10)

#     phase = spec[:, :, 1, :, :] 
#     phase = torch.clamp(phase, -1, 1) * torch.pi

#     wav_spec = amp * (torch.cos(phase) + 1j * torch.sin(phase))
#     # wav_spec = torch.polar(amp, phase)
#     wav_spec = wav_spec.view(B*N, 256, -1)
#     waveforms = torch.istft(
#         wav_spec, 
#         n_fft=510, 
#         hop_length=256, 
#         win_length=510, 
#         window=torch.hann_window(510).to(device), 
#         center=True, 
#         normalized=False)
#     waveforms = waveforms.view(B, N, -1)
#     return waveforms




spec_factor=0.15
spec_factor_2 = torch.tensor(
    [[2.1248443126678467, 1.48661208152771], [2.170947551727295, 1.5203202962875366], [2.1222591400146484, 1.4931153059005737], [2.1758151054382324, 1.5207386016845703]]
) # [4, 2]
spec_factor_2 = torch.tensor(
    [[2.2, 1.3], [2.2, 1.6], [2.2, 1.6], [2.2, 1.6]]
)
# spec_factor_2 = torch.tensor([[1., 1.], [1., 1.], [1., 1.], [1., 1.]]) # [4, 2]

spec_abs_exponent=0.5
def audio2spec(waveforms):
    """
    audio: [batch_size, N, T]
    return: [batch_size, N, 2, 256, T']
    """
    assert waveforms.dim() == 3 
    B, N, L = waveforms.shape
    device = waveforms.device

    wav_spec = torch.stft(
        waveforms.view(-1, waveforms.size(-1)), 
        n_fft=510, 
        hop_length=256, 
        win_length=510, 
        window=torch.hann_window(510).to(device), 
        return_complex=True, center=True, 
        normalized=False)
    _, H, W = wav_spec.shape
    wav_spec = wav_spec.reshape(B, N, H, W)
    
    e = spec_abs_exponent
    spec = wav_spec.abs()**e * torch.exp(1j * wav_spec.angle())
    spec = spec * spec_factor
    output = torch.view_as_real(spec).permute(0, 1, 4, 2, 3)
    output = output / spec_factor_2[:N].reshape(1, N, 2, 1, 1).to(device)
    output = torch.clamp(output, -1, 1)

    return output

def spec2audio(spec):
    """
    spec: [batch_size, N, 2, 256, T']
    return: [batch_size, N, T]
    """
    device = spec.device
    if spec.dim() == 4:
        spec = spec.reshape(spec.size(0), -1, 2, spec.size(-2), spec.size(-1))
    B, N, _, _, _ = spec.shape

    spec = torch.clamp(spec, -1, 1)

    spec = spec * spec_factor_2[:N].view(1, N, 2, 1, 1).to(device)

    spec = spec.permute(0, 1, 3, 4, 2)
    spec = torch.view_as_complex(spec.contiguous())
    spec = spec / spec_factor
    if spec_abs_exponent != 1:
        e = spec_abs_exponent
        spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())

    # wav_spec = torch.polar(amp, phase)
    wav_spec = spec.view(B*N, 256, -1)
    waveforms = torch.istft(
        wav_spec, 
        n_fft=510, 
        hop_length=256, 
        win_length=510, 
        window=torch.hann_window(510).to(device), 
        center=True, 
        normalized=False)
    waveforms = waveforms.view(B, N, -1)
    return waveforms

