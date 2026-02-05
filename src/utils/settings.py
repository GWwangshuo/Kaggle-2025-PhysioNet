import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    raw_data_dir: Path
    processed_data_dir: Path
    model_dir: Path
    submission_dir: Path


def load_settings(json_path: str = 'SETTINGS.json') -> Settings:
    with open(json_path, 'r') as f:
        data = json.load(f)
        settings = Settings(
            raw_data_dir=Path(data['RAW_DATA_DIR']),
            processed_data_dir=Path(data['PROCESSED_DATA_DIR']),
            model_dir=Path(data['MODEL_DIR']),
            submission_dir=Path(data['SUBMISSION_DIR'])
        )
    return settings


if __name__ == '__main__':
    settings = load_settings()
    print(settings)