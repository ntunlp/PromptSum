from transformers import T5Tokenizer, T5ForConditionalGeneration


model_name = "google/t5-v1_1-base"
cache_path = "/data/mathieu/hf_models/t5-v1-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_path)
t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_path)

sequence = "my dog is under the table"
sequence_tok = tokenizer(sequence, return_tensors="pt")
labels = "have you"
labels_tok = tokenizer(labels, return_tensors="pt")
prompt = "summarize"
prompt_tok = tokenizer(prompt, return_tensors="pt")
print(sequence_tok["input_ids"].shape, sequence_tok["attention_mask"].shape, prompt_tok["input_ids"].shape, labels_tok["input_ids"].shape)
outs = t5model.generate(
    input_ids = sequence_tok["input_ids"], 
    attention_mask = sequence_tok["attention_mask"],
    #labels = labels_tok["input_ids"],
    #decoder_input_ids = labels_tok["input_ids"],
    #min_length = 20
)
print(outs)
out = tokenizer.batch_decode(outs)
print(out)

