import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Some graph reductions have no deterministic CUDA implementation.
    torch.use_deterministic_algorithms(True, warn_only=True)
