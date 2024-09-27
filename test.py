from transformers import AutoConfig, AutoTokenizer
from model_arch import KeallmForConditionalGeneration, KeallmConfig

# kge_config._name_or_path= "bert-base-uncased"

text_config = AutoConfig.from_pretrained("Qwen/Qwen2-1.5B-Instruct", device_map="auto")
kge_config = AutoConfig.from_pretrained("ledong0110/FB15k-237-KGE-Roberta-Base", device_map="auto")
keallm_config = KeallmConfig.from_kge_text_configs(kge_config=kge_config, text_config=text_config)
model = KeallmForConditionalGeneration(keallm_config)
model.push_to_hub("ledong0110/KEALLM-Qwen2-Roberta-1.5B")

# model = KeallmForConditionalGeneration.from_pretrained("KEALLM-Roberta-Base-7B", device_map="auto")

kge_tokenizer = AutoTokenizer.from_pretrained("ledong0110/FB15k-237-KGE-Roberta-Base")
text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

kge_text = "The US [PAD] [SEP] [inverse] is president of [PAD] [SEP] [MASK]"
kge_input_ids = kge_tokenizer(kge_text, return_tensors="pt").input_ids.to(model.device)
text_input = text_tokenizer("Who is the president of US", return_tensors="pt").to(model.device)

a = model.generate(kge_input_ids, **text_input, max_new_tokens=15)

print(text_tokenizer.batch_decode(a))

labels = text_tokenizer("The president of US is Dong Le", return_tensors="pt").input_ids
output = model(kge_input_ids, **text_input, labels=labels)

