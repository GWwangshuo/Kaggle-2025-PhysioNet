import numpy as np
from scipy.signal import resample_poly, resample
from scipy.interpolate import CubicSpline
import torch
import torch.nn.functional as F
from glob import glob
import pandas as pd
from tqdm.auto import tqdm
import torchaudio
from resampler import Resampler
import matplotlib.pyplot as plt


def read_truth_series(truth_df):

    truth_df["II-rhythm"] = truth_df["II"]
    truth_df.loc[truth_df["I"].isna(), "II"] = np.nan
    truth_df.fillna(0, inplace=True)

    series0 = (truth_df["I"] + truth_df["aVR"] + truth_df["V1"] + truth_df["V4"]).values
    series1 = (truth_df["II"] + truth_df["aVL"] + truth_df["V2"] + truth_df["V5"]).values
    series2 = (truth_df["III"] + truth_df["aVF"] + truth_df["V3"] + truth_df["V6"]).values
    series3 = (truth_df["II-rhythm"]).values
    truth_df["series0"] = series0
    truth_df["series1"] = series1
    truth_df["series2"] = series2
    truth_df["series3"] = series3
    return truth_df
    

def resample_poly_safe(x, target_len, axis=-1, pad_len=None):
    L = x.shape[axis]
    if pad_len is None:
        pad_len = 10
    if L <= pad_len:
        pad_len = L // 2

    pad_front = np.flip(np.take(x, np.arange(pad_len), axis=axis), axis=axis)
    pad_back  = np.flip(np.take(x, np.arange(L-pad_len, L), axis=axis), axis=axis)
    x_pad = np.concatenate([pad_front, x, pad_back], axis=axis)

    y_pad = resample_poly(x_pad, up=target_len, down=L, axis=axis)

    start = int(pad_len * target_len / L)
    end   = start + target_len
    y = np.take(y_pad, np.arange(start, end), axis=axis)
    return y

def resample_torch(x, num, dim=-1):
    """基于 FFT 的 PyTorch 重采样"""
    dim = (x.dim() + dim) if dim < 0 else dim
    X = torch.fft.fft(x, dim=dim)
    Nx = X.shape[dim]

    sl = [slice(None)] * X.ndim
    newshape = list(X.shape)
    newshape[dim] = num
    Y = torch.zeros(newshape, dtype=X.dtype, device=X.device)

    N = min(num, Nx)
    sl[dim] = slice(0, (N + 1) // 2)
    Y[sl] = X[sl]
    sl[dim] = slice(-(N - 1) // 2, None)
    Y[sl] = X[sl]

    if N % 2 == 0:
        if N < Nx:
            sl[dim] = slice(N//2, N//2+1)
            Y[sl] += X[sl]
        elif N < num:
            sl[dim] = slice(num-N//2, num-N//2+1)
            Y[sl] /= 2
            temp = Y[sl]
            sl[dim] = slice(N//2, N//2+1)
            Y[sl] = temp

    y = torch.fft.ifft(Y, dim=dim).real * (float(num) / float(Nx))
    return y

def resampling(x, target_len=2500, method='resample_poly'):
    """
    x: np.ndarray, shape [C, L]
    target_len: int
    method: str, 'resample_poly' | 'resample' | 'cubic' | 'linear' | 'torch_linear' | 'torch_cubic' | 'torchaudio' | 'torch_resampler'
    """
    C, L = x.shape

    if L == target_len:
        return x

    if method == 'resample_poly':
        y = resample_poly_safe(x, target_len, axis=1)

    elif method == 'resample':
        y = resample(x, target_len, axis=1)

    elif method == 'cubic':
        x_old = np.linspace(0, 1, L)
        x_new = np.linspace(0, 1, target_len)
        y = np.zeros((C, target_len), dtype=x.dtype)
        for c in range(C):
            cs = CubicSpline(x_old, x[c])
            y[c] = cs(x_new)

    elif method == 'linear':
        x_old = np.linspace(0, 1, L)
        x_new = np.linspace(0, 1, target_len)
        y = np.zeros((C, target_len), dtype=x.dtype)
        for c in range(C):
            y[c] = np.interp(x_new, x_old, x[c])

    elif method in ['torch_linear', 'torch_cubic']:
        mode = 'linear' if method == 'torch_linear' else 'bicubic'
        x_tensor = torch.from_numpy(x).unsqueeze(0)  # [1, C, L]
        if mode == 'bicubic' and C == 1:
            x_tensor = x_tensor.unsqueeze(-1)  # [1, 1, L, 1]
            y_tensor = F.interpolate(x_tensor, size=(target_len, 1), mode='bicubic', align_corners=False)
            y_tensor = y_tensor.squeeze(-1)
        else:
            y_tensor = F.interpolate(x_tensor, size=target_len, mode=mode, align_corners=False)
        y = y_tensor.squeeze(0).numpy()

    elif method == 'torchaudio':
        orig_freq_auto = L
        new_freq_auto = target_len
        x_tensor = torch.from_numpy(x).float()  # [C, L]
        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq_auto, new_freq=new_freq_auto)
        y_tensor = resampler(x_tensor)  # [C, target_len]
        y = y_tensor.numpy()

    elif method == 'torch_resampler':
        # 使用 Resampler 类
        input_sr_auto = L
        output_sr_auto = target_len
        x_tensor = torch.from_numpy(x).float()  # [C, L]
        resampler = Resampler(input_sr=input_sr_auto, output_sr=output_sr_auto, dtype=torch.float32)
        y_tensor = resampler(x_tensor)  # [C, target_len]
        y = y_tensor.numpy()
    elif method == 'fft':
        x_tensor = torch.from_numpy(x).float()
        y_tensor = resample_torch(x_tensor, target_len, dim=-1)
        y = y_tensor.numpy()

    else:
        raise ValueError(f"Unknown method: {method}")

    return y


def np_snr(predict, truth, eps=1e-7):
    signal = (truth ** 2).sum()
    noise  = ((predict - truth) ** 2).sum()
    snr = signal / (noise + eps)
    snr_db = 10 * np.log10(snr + eps)
    return snr_db


if __name__ == '__main__':
    data_root = "/data/gavin/challenges/kaggle/2025/PhysioNet/gavin_baseline/input/physionet-ecg-image-digitization/train"
    csv_files = glob(f'{data_root}/*/*.csv')
   
    target_len = 2560 # 5000, 5120, 10000, 10250
    methods = ['resample', 'torch_linear']

    snr_records = {m: [] for m in methods}

    tmp = []
    for csv_path in tqdm(csv_files):
        df = pd.read_csv(csv_path)
        truth_df = read_truth_series(df)
        x = truth_df[["series0", "series1", "series2", "series3"]].values.T
        L = x.shape[1]
        
        for m in methods:
            y = resampling(x, target_len=target_len, method=m)
            y_up = resampling(y, target_len=L, method=m)
            snr_value = np_snr(y_up, x)
            
            snr_records[m].append(snr_value)

    for m in methods:
        snr_mean = np.mean(snr_records[m])
        print(f'{m}: snr_value_mean = {snr_mean:.4f}')
        
    m0 = methods[0]
    idx = np.argsort(snr_records[m0])

    sorted_snr = {m: np.array(snr_records[m])[idx] for m in methods}

    plt.figure(figsize=(8, 5))
    for m in methods:
        # plt.plot(sorted_snr[m], label=m, marker='o')
        snrs = sorted_snr[m]
        plt.scatter(range(len(snrs)), snrs, label=m, alpha=0.6)
        
    plt.xlabel("Sample index (sorted by {})".format(m0))
    plt.ylabel("SNR")
    plt.title("SNR Comparison of Methods")
    plt.legend()
    plt.grid(True)
    plt.show()

    for method, snrs in snr_records.items():
        print(f"{method} average SNR: {np.mean(snrs):.4f} dB")
