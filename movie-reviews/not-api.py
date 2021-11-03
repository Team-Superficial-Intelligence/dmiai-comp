from ml.emily import Emily
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

emily = Emily()

tok = emily.superfuntime()
