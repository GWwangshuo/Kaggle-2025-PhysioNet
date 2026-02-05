import os
import zipfile

import kagglehub
from kaggle.api.kaggle_api_extended import KaggleApi

from src.utils.settings import load_settings


def main():
    settings = load_settings()
    raw_data_dir = settings.raw_data_dir
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    processed_data_dir = settings.processed_data_dir
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    datasets = [
        {
            'type': 'competition',
            'name': 'physionet-ecg-image-digitization',
            'extract_dir': raw_data_dir / 'physionet-ecg-image-digitization'
        },
        {
            'type': 'dataset',
            'name': 'sjtuwangshuo/2nd-stage-data',
            'extract_dir': processed_data_dir
        }
        
    ]

    for item in datasets:
        print(f"Downloading competition data {item['name']}")
        
        if item['type'] == 'dataset':
            api.dataset_download_files(item['name'], path=str(raw_data_dir), unzip=False, quiet=False)
        elif item['type'] == 'competition':
            api.competition_download_files(item['name'], path=str(raw_data_dir), quiet=False)
        
        zip_filename = item['name'] + '.zip'
       
        zip_file = raw_data_dir / zip_filename
        extract_dir = item['extract_dir']

        print(f'Extracting {zip_file} to {extract_dir}')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f'Removing {zip_file}')
        os.remove(zip_file)


if __name__ == '__main__':
    main()