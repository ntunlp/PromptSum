import os
import re
import torch
from tqdm import tqdm
import tensorflow as tf
from transformers import PegasusForConditionalGeneration

# load pytorch model
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
name = list(model.state_dict().keys())
for n in name:
    print(n)
layer = "model.encoder.layers.15.self_attn_layer_norm.bias"
print(model.state_dict()[layer].sum())
#raise Exception

# load tensorflow checkpoint
ckpt_path = "frost_ckpt/frost_ckpt"
tf_path = os.path.abspath(ckpt_path)
init_vars = tf.train.list_variables(tf_path)
tf_vars = {}
for name, shape in init_vars:
    if "Adafactor" in name:
        continue
    if name == "global_step":
        continue
    array = tf.train.load_variable(tf_path, name)
    if len(array.shape) > 0:
        array = torch.from_numpy(array)
    else:
        array = torch(array)
    array = array.squeeze()
    tf_vars[name] = array
print(len(tf_vars))

# assign tf weights into the pt state dict
d = {}

# encoder & decoder layers
n_layers = 16
for network in ["encoder", "decoder"]:
    for i in range(n_layers):
        # self-attention
        for x in ["k", "v", "q"]:
            pt_name = "model.{}.layers.{}.self_attn.{}_proj.weight".format(network, i, x)
            tf_name = "{}/layer_{}/self_attention/{}_proj/kernel".format(network, i, x)
            d[pt_name] = tf_vars[tf_name]
        pt_name = "model.{}.layers.{}.self_attn.out_proj.weight".format(network, i)
        tf_name = "{}/layer_{}/self_attention/output_proj/kernel".format(network, i)
        d[pt_name] = tf_vars[tf_name]
        pt_name = "model.{}.layers.{}.self_attn_layer_norm.weight".format(network, i)
        tf_name = "{}/layer_{}/self_attention/LayerNorm/gamma".format(network, i)
        d[pt_name] = tf_vars[tf_name]
        pt_name = "model.{}.layers.{}.self_attn_layer_norm.bias".format(network, i)
        tf_name = "{}/layer_{}/self_attention/LayerNorm/beta".format(network, i)
        d[pt_name] = tf_vars[tf_name]
        # 1st dense layer 
        pt_name = "model.{}.layers.{}.fc1.weight".format(network, i)
        tf_name = "{}/layer_{}/ffn/dense/kernel".format(network, i)
        d[pt_name] = torch.transpose(tf_vars[tf_name], 0, 1)
        pt_name = "model.{}.layers.{}.fc1.bias".format(network, i)
        tf_name = "{}/layer_{}/ffn/dense/bias".format(network, i)
        d[pt_name] = tf_vars[tf_name]
        # 2nd dense layer
        pt_name = "model.{}.layers.{}.fc2.weight".format(network, i)
        tf_name = "{}/layer_{}/ffn/dense_1/kernel".format(network, i)
        d[pt_name] = torch.transpose(tf_vars[tf_name], 0, 1)
        pt_name = "model.{}.layers.{}.fc2.bias".format(network, i)
        tf_name = "{}/layer_{}/ffn/dense_1/bias".format(network, i)
        d[pt_name] = tf_vars[tf_name]
        # final_layer_norm
        pt_name = "model.{}.layers.{}.final_layer_norm.weight".format(network, i)
        tf_name = "{}/layer_{}/ffn/LayerNorm/gamma".format(network, i)
        d[pt_name] = tf_vars[tf_name]
        pt_name = "model.{}.layers.{}.final_layer_norm.bias".format(network, i)
        tf_name = "{}/layer_{}/ffn/LayerNorm/beta".format(network, i)
        d[pt_name] = tf_vars[tf_name]
        # cross-attention (decoder-only)
        if network == "decoder":
            for x in ["k", "v", "q"]:
                pt_name = "model.{}.layers.{}.encoder_attn.{}_proj.weight".format(network, i, x)
                tf_name = "{}/layer_{}/memory_attention/{}_proj/kernel".format(network, i, x)
                d[pt_name] = tf_vars[tf_name]
            pt_name = "model.{}.layers.{}.encoder_attn.out_proj.weight".format(network, i)
            tf_name = "{}/layer_{}/memory_attention/output_proj/kernel".format(network, i)
            d[pt_name] = tf_vars[tf_name]
            pt_name = "model.{}.layers.{}.encoder_attn_layer_norm.weight".format(network, i)
            tf_name = "{}/layer_{}/memory_attention/LayerNorm/gamma".format(network, i)
            d[pt_name] = tf_vars[tf_name]
            pt_name = "model.{}.layers.{}.encoder_attn_layer_norm.bias".format(network, i)
            tf_name = "{}/layer_{}/memory_attention/LayerNorm/beta".format(network, i)
            d[pt_name] = tf_vars[tf_name]

# embeddings
pt_name = "model.shared.weight"
tf_name = "embeddings/weights"
d[pt_name] = tf_vars[tf_name]
pt_name = "model.encoder.embed_tokens.weight"
d[pt_name] = tf_vars[tf_name]
pt_name = "model.decoder.embed_tokens.weight"
d[pt_name] = tf_vars[tf_name]
pt_name = "lm_head.weight"
d[pt_name] = tf_vars[tf_name]

# layer norms
pt_name = "model.encoder.layer_norm.weight"
tf_name = "encoder/LayerNorm/gamma"
d[pt_name] = tf_vars[tf_name]
pt_name = "model.encoder.layer_norm.bias"
tf_name = "encoder/LayerNorm/beta"
d[pt_name] = tf_vars[tf_name]
pt_name = "model.decoder.layer_norm.weight"
tf_name = "decoder/LayerNorm/gamma"
d[pt_name] = tf_vars[tf_name]
pt_name = "model.decoder.layer_norm.bias"
tf_name = "decoder/LayerNorm/beta"
d[pt_name] = tf_vars[tf_name]

print(model.state_dict()["model.encoder.embed_positions.weight"].shape)
print(model.state_dict()["model.decoder.embed_positions.weight"].shape)
print(model.state_dict()["final_logits_bias"].shape)
print(model.state_dict()["lm_head.weight"].sum())
print(model.state_dict()["model.encoder.embed_tokens.weight"].sum())
print(model.state_dict()["model.decoder.embed_tokens.weight"].sum())
print(model.state_dict()["model.shared.weight"].sum())

for k in model.state_dict().keys():
    if not(k in d.keys()):
        print("Missing from d: {}".format(k), model.state_dict()[k].sum())
        d[k] = model.state_dict()[k]

for k in d.keys():
    if not(k in model.state_dict().keys()):
        print("In d but not in state_dict: {}".format(k))

model.load_state_dict(d)
print("loaded weights!")

