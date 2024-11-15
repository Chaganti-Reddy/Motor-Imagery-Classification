from pathlib import Path

ROOT_DIR      = Path(__file__).resolve().parent.parent
DATASET_DIR   = ROOT_DIR / "dataset"
OUTPUTS_DIR   = ROOT_DIR / "outputs"
SYNTHETIC_DIR = OUTPUTS_DIR / "synthetic"
MODELS_DIR    = OUTPUTS_DIR / "models"
FIGURES_DIR   = OUTPUTS_DIR / "figures"
METRICS_DIR   = ROOT_DIR / "metrics"

for _d in [SYNTHETIC_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

N_SUBJECTS   = 9
N_CLASSES    = 4
CLASS_NAMES  = ["Left Hand", "Right Hand", "Both Feet", "Tongue"]
EVENT_IDS = {769: 0, 770: 1, 771: 2, 772: 3}
ELECTRODE_NAMES = ["EEG-Fz", "EEG-C3", "EEG-Cz", "EEG-C4", "EEG-Oz"]
N_CHANNELS      = 5
SAMPLING_RATE = 250 
TMIN          = 0.0 
TMAX          = 4.0 
N_TIME_SUBSAMPLE = 375
N_CWT_SCALES  = 50
CWT_WAVELET   = "morl"
CWT_SCALES    = list(range(1, N_CWT_SCALES + 1)) 
TRIAL_SHAPE = (N_CWT_SCALES, N_TIME_SUBSAMPLE, N_CHANNELS) 
LATENT_DIM      = 100
WGAN_BATCH_SIZE = 100
WGAN_EPOCHS     = 300
N_CRITIC        = 5   
GP_LAMBDA       = 10   
WGAN_LR         = 1e-4  
WGAN_BETA1      = 0.0  
WGAN_BETA2      = 0.9 
N_SYNTHETIC_PER_CLASS = 100
CNN_LR          = 1e-4
CNN_EPOCHS      = 70
CNN_BATCH_SIZE  = 32
CNN_DROPOUT     = 0.5
CNN_L2          = 0.01
CNN_DENSE_UNITS = 750
CNN_FILTERS     = 32
CNN_KERNEL_SIZE = (7, 7)
CNN_VAL_SPLIT   = 0.20  
ROTATION_ANGLE   = 180  
NOISE_ALPHA_LOW  = 0.1  
NOISE_ALPHA_HIGH = 0.5 
RANDOM_SEED = 42
