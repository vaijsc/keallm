from torch.utils.data import Dataset
import json

class KBQADataset(Dataset):
    def __init__(self, args, mode):
        super().__init__()
        self.args = args
        self.mode = mode
        self.data = self.loadData()
    
    def loadData(self):
        with open(f"./{self.args.dataset_dir}/{self.args.dataset}/{self.args.k_hop}-hop/{self.mode}.json") as file:
            data = [json.loads(_) for _ in file.readlines()]
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]