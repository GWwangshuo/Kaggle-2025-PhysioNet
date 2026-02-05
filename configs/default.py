import os

class Config:
    # --- Path configuration (corresponds to SETTINGS.json) ---
    DATA_PATH = 'data/raw/physionet-ecg-image-digitization'
    SECOND_STAGE_DATA_PATH = 'data/processed/2nd-stage-data'
    OUTPUT_DIR = 'models/trained_models'
    PRETRAINED_WEIGHTS = "models/pretrained/unet_weights_07072025.pt"
    
    # --- Model architecture parameters ---
    MODEL_PARAMS = {
        "num_in_channels": 3,
        "num_out_channels": 4,
        "dims": [32, 64, 128, 256, 320, 320, 320, 320],
        "depth": 2
    }

    # --- Training hyperparameters ---
    START_EPOCH = 1
    EPOCHS = 200
    WARMUP_EPOCHS = 5
    BATCH_SIZE = 1
    LR = 5e-4
    MIN_LR = 0.0
    LR_SCHEDULE = 'cosine'
    WEIGHT_DECAY = 0.01
    FOLD = 'all'
    
    # --- System parameters ---
    SEED = 0
    NUM_WORKERS = 12
    PIN_MEM = True
    DEVICE = 'cuda'
    
    # --- Logging and checkpointing ---
    SAVE_LAST_FREQ = 5
    LOG_FREQ = 100
    
    # --- Distributed training ---
    WORLD_SIZE = 1
    LOCAL_RANK = -1
    DIST_URL = 'env://'

# Instantiate configuration
cfg = Config()