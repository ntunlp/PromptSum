from transformers import T5Tokenizer, T5ForConditionalGeneration


model_name = "google/t5-v1_1-base"
cache_path = "/data/mathieu/hf_models/t5-v1-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_path)
t5model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_path)

sequence = "my dog is under the table"
sequence_tok = tokenizer(sequence, return_tensors="pt")
prompt = "summarize"
prompt_tok = tokenizer(prompt, return_tensors="pt")
labels = "have you"
labels_tok = tokenizer(labels, return_tensors="pt")
print(sequence_tok["input_ids"].shape, prompt_tok["input_ids"].shape, labels_tok["input_ids"].shape)

labels_tok["input_ids"] = torch.cat((prompt_tok["input_ids"], labels_tok["input_ids"]), 0)
# labels_tok["attention_mask"] = torch.cat((prompt_tok["attention_mask"], labels_tok["attention_mask"]), 0)

outs = t5model.forward(
    input_ids = sequence_tok["input_ids"], 
    attention_mask = sequence_tok["attention_mask"],
    labels = labels_tok["input_ids"],
    #decoder_input_ids = labels_tok["input_ids"],
    #min_length = 20
)
logits = outs["logits"]
print(logits.shape)

