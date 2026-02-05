import cv2
import random
import torch
import numpy as np
import pandas as pd
from typing import Any, Tuple
from scipy.signal import resample


class ECGScanDataset(torch.utils.data.Dataset[Any]):
    def __init__(
        self, 
        raw_data_path: str, 
        second_stage_data_path: str,
        transform: Any = None, 
        mode: str = 'train'
    ) -> None:
        self.raw_data_path = raw_data_path
        self.second_stage_data_path = second_stage_data_path
        self.transform = transform
        self.mode = mode

        df = pd.read_csv(f'{raw_data_path}/train.csv')

        sample_ids = df['id'].tolist()
            
        type_ids = ['0003', '0004', '0005', '0006', '0009', '0010', '0011', '0012']

        self.ecg_scan_files = [
            f'{image_id}-{type_id}'
            for image_id in sample_ids
            for type_id in type_ids
        ]
        
    def _load_scan(self, file: str):
        scan = cv2.imread(file, cv2.IMREAD_COLOR_RGB)
        return scan
    
    def __len__(self) -> int:
        return len(self.ecg_scan_files)
    
    def read_truth_series(self, sample_id):
        image_id = sample_id.split("-")[0]
    
        truth_df = pd.read_csv(f"{self.raw_data_path}/train/{image_id}/{image_id}.csv")
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
    
    def resample_series(
        self,
        series: np.ndarray,
        target_len: int
    ):
        """
        Polyphase FIR resampling with boundary artifact suppression.

        Args:
            series: (C, L) numpy array
            target_len: target length after resampling
            beta: Kaiser window beta
            pad_ratio: padding ratio w.r.t original length

        Returns:
            out: (C, target_len) float32
        """
        C, L = series.shape

        if L == target_len:
            return series.astype(np.float32)

        out = resample(series, target_len, axis=1)

        return out.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    
        scan_id = self.ecg_scan_files[idx]
        scan_path = f'{self.second_stage_data_path}/recified_images/{scan_id}.png'
        
        scan = self._load_scan(scan_path)
        
        truth_df = self.read_truth_series(scan_id)
        truth_series = truth_df[["series0", "series1", "series2", "series3",]].values.T
        L = truth_series.shape[1]
         
        x0, x1 = 0, 2176
        y0, y1 = 0, 1696
        t0, t1 = 118, 2080
        
        crop = scan[y0:y1, x0:x1, :][:, t0:t1, :]
        crop = (crop / 255.0).astype(np.float32)
        
        if self.transform:
            transformed = self.transform(
                image=crop,
            )
            crop = transformed['image']
        
        crop = np.transpose(crop, (2, 0, 1)).astype(np.float32)
        
        ecg_mv = {
            str(fs): self.resample_series(truth_series, target_len=fs)
            for fs in [2500, 2560, 5000, 5120, 10000, 10250]
        }
        
        if random.random() < 0.5:
            crop = crop[:, :, ::-1].copy()          # (C, H, W)
            ecg_mv = {k: v[:, ::-1].copy()          # (C, L)
                    for k, v in ecg_mv.items()}
        
        ecg_mv['fs'] = L
                
        out = {
            "scan": crop,
            "ecg_mv": ecg_mv
        }

        return out
    
    
if __name__ == '__main__':
    from src.utils.transforms import build_transforms
    dataset = ECGScanDataset(
        data_path="input/physionet-ecg-image-digitization",
        transform=build_transforms()
    )
    dataset[0]
    
    
