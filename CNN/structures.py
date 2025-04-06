import torch

def time_reversal(x):
    return torch.flip(x, dims=[-1])  # flip along width (time)

def frequency_inversion(x):
    return torch.flip(x, dims=[-2])  # flip along height (frequency)

def tf_transforms(x):
    return [
        x,
        time_reversal(x),
        frequency_inversion(x),
        frequency_inversion(time_reversal(x))
    ]
