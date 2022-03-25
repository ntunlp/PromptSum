from transformers import T5Tokenizer, T5ForConditionalGeneration,


model_name = "google/t5-v1_1-base"
cache_path = "/data/mathieu/hf_models/t5-v1-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_path)
t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_path)

sequence = "my dog is under the table"
sequence_tok = tokenizer(sequence, return_tensors="pt")
outs = t5model(input_ids = sequence_tok["input_ids"], attention_mask = sequence_tok["attention_mask"])
print(out)


