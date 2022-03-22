'''Convert tensorflow lm adapted t5 model to pytorch (huggingface) 
'''
from transformers import T5ForConditionalGeneration
import transformers 
def convert(model_type, ckpt_path, save_dir):
    model = T5ForConditionalGeneration.from_pretrained(model_type)
    lm_adapted_model = transformers.models.t5.modeling_t5.load_tf_weights_in_t5(model, None, ckpt_path)
    lm_adapted_model.save_pretrained(save_dir)
    print(f"Saved {model_type} to {save_dir}")

model_types = ['google/t5-v1_1-small','google/t5-v1_1-base','google/t5-v1_1-large']
dir_template = '/export/home/prompting/lm_adapted_models/t5.1.1.lm100k.{}/'
save_dirs = [dir_template.format('small'), dir_template.format('base'), dir_template.format('large')]
ckpt_paths = [save_dir+'model.ckpt-1100000' for save_dir in save_dirs]
for model_type, ckpt_path, save_dir in zip(model_types, ckpt_paths, save_dirs):
    convert(model_type, ckpt_path, save_dir)