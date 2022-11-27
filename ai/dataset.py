"""
DataModules
"""
from torch.utils.data import Dataset, DataLoader


class PatentDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
