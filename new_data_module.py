from torch.utils.data import DataLoader
import argparse
from base_qa_data import KBQADataset
import time


class Config(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, val):
        self[name] = val

BATCH_SIZE = 8
NUM_WORKERS = 8   
        
class QADataModule:
    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = Config(vars(args)) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", NUM_WORKERS)


        # base setting
        self.entity2text = self.get_entity_to_text()
        self.relation2text = self.get_relation_to_text()
        self.num_entity = len(self.entity2text.keys())
        self.num_relation = len(self.relation2text.keys())
    
    def get_entity_to_text(self):
        entity2text = {}
        with open(f"./{self.args.dataset_dir}/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                id_, text = line.strip().split("\t")
                entity2text[int(id_)] = text
        return entity2text
    
    def get_relation_to_text(self):
        relation2text = {}
        with open(f"./{self.args.dataset_dir}/{self.args.dataset}/relation2text.txt") as file:
            for line in file.readlines():
                id_, text = line.strip().split("\t")
                relation2text[int(id_)] = text
        
        return relation2text
    
    def setup(self, stage=None):
        now_time = time.time()
        print("setup data for each process...")
        if stage == "fit":
            self.data_train = KBQADataset(self.args, mode="train")
            self.data_val = KBQADataset(self.args, mode="dev")
        else:
            self.data_test = KBQADataset(self.args, mode="test")
        
        
        entity2text = []
        with open(f"{self.args.dataset_dir}/{self.args.dataset}/entity2text.txt") as file:
            for line in file.readlines():
                line = line.strip().split("\t")[1]
                entity2text.append(line)
        
        self.entity_strings = entity2text
        # self.tokenized_entities = self.tokenizer(entity2text, padding='max_length', truncation=True, max_length=self.args.max_entity_length, return_tensors="pt")

        print("finished data processing... costing {}s...".format(time.time() - now_time))
    
    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.num_workers, 
            collate_fn=partial(self.collate_fn, mode="train"), pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="dev"), pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.args.eval_batch_size, num_workers=self.args.num_workers, collate_fn=partial(self.collate_fn, mode="test"), pin_memory=True, drop_last=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_dir", type=str, default="dataset")
    argparser.add_argument("--dataset", type=str, default="MetaQA")
    argparser.add_argument("--k_hop", type=int, default=1)
    argparser.add_argument("--max_entity_length", type=int, default=32)
    argparser.add_argument("--max_relation_length", type=int, default=32)
    argparser.add_argument("--max_question_length", type=int, default=64)
    argparser.add_argument("--batch_size", type=int, default=8)
    argparser.add_argument("--eval_batch_size", type=int, default=32)
    argparser.add_argument("--num_workers", type=int, default=4)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--max_epochs", type=int, default=10)
    args = argparser.parse_args()
    data = QADataModule(args)
    data.setup()
    print(data.data_test[0])