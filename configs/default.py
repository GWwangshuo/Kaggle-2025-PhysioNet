import os

class Config:
    # --- 路径配置 (对应 SETTINGS.json) ---
    DATA_PATH = 'data/raw/physionet-ecg-image-digitization'
    SECOND_STAGE_DATA_PATH = 'data/processed/2nd-stage-data'
    OUTPUT_DIR = 'models/trained_models'
    PRETRAINED_WEIGHTS = "models/pretrained/unet_weights_07072025.pt"
    
    # --- 模型架构参数 ---
    MODEL_PARAMS = {
        "num_in_channels": 3,
        "num_out_channels": 4,
        "dims": [32, 64, 128, 256, 320, 320, 320, 320],
        "depth": 2
    }

    # --- 训练超参数 ---
    START_EPOCH = 1
    EPOCHS = 200
    WARMUP_EPOCHS = 5
    BATCH_SIZE = 1
    LR = 5e-4
    MIN_LR = 0.0
    LR_SCHEDULE = 'cosine'
    WEIGHT_DECAY = 0.01
    FOLD = 'all'
    
    # --- 系统参数 ---
    SEED = 0
    NUM_WORKERS = 12
    PIN_MEM = True
    DEVICE = 'cuda'
    
    # --- 日志与保存 ---
    SAVE_LAST_FREQ = 5
    LOG_FREQ = 100
    
    # --- 分布式训练 ---
    WORLD_SIZE = 1
    LOCAL_RANK = -1
    DIST_URL = 'env://'

# 实例化
cfg = Config()