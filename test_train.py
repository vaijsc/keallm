from data_module.data_module import FB15k237DataModule
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--punc_split", type=str, default=" , ", help="Punctuation to split on")
FB15k237DataModule.add_to_argparse(parser)
args = parser.parse_args()
data_module = FB15k237DataModule(args)
data_module.setup()
train_data = data_module.val_dataset
print(train_data[0])
# test_data = data_module.test_dataloader()
# for i in iter(test_data):
#     print(i)
#     break