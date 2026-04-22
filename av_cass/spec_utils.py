import torch

spec_factor=0.15
spec_factor_2 = torch.tensor(
    [[2.1248443126678467, 1.48661208152771], [2.170947551727295, 1.5203202962875366], [2.1222591400146484, 1.4931153059005737], [2.1758151054382324, 1.5207386016845703]]
) # [4, 2]
spec_factor_2 = torch.tensor(
    [[2.2, 1.3], [2.2, 1.6], [2.2, 1.6], [2.2, 1.6]]
)


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

