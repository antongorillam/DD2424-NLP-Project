import torch
import torch.nn as nn

torch.device = torch.device("cude" if torch.cuda.is_available() else "cpu")