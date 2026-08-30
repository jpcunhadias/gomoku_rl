import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """
    Set the global random seed for all relevant libraries to ensure reproducibility.

    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[Seeding] Global seed set to {seed}")
