import torch
import os
import sys
sys.path.append("../support_files/")
import numpy as np
import random
import csv
import pickle
import pickle5
import spacy
import tensorflow as tf



class VirtualList(object):
    def __init__(self, dataset, field):
        self.dataset = dataset
        self.field = field 
    
    def __getitem__(self, idx):
        return self.dataset[idx][self.field]

    def __len__(self):
        return len(self.dataset)

class Nop(object):
    def nop(*args, **kw): pass
    def __getattr__(self, _): return self.nop
        

def seed_everything(args):
    seed = int(args.seed)
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def getfewshot(inpath,outpath,fewshotnum):
    ###read from inpath
    intrain = inpath + "/train.txt"
    invalid = inpath + "/valid.txt"
    intest = inpath + "/test.txt"
    alltrainres = []
    allvalidres = []
    alltestres = []

    f = open(intrain,'r')
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        alltrainres.append(oneline)
    f.close()

    f = open(invalid, 'r')
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        allvalidres.append(oneline)
    f.close()

    f = open(intest, 'r')
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        alltestres.append(oneline)
    f.close()

    ######select few shot for train valid and test
    ###outpath
    fewtrainname = outpath + "/train.txt"
    fewvalidname = outpath + "/valid.txt"
    fewtestname = outpath + "/test.txt"

    tousetrainres = random.sample(alltrainres, fewshotnum)
    tousevalidres = random.sample(allvalidres, fewshotnum)
    testnum = 1000
    tousetestres = random.sample(alltestres, testnum)

    f = open(fewtrainname,'w')
    for one in tousetrainres:
        f.write(one + "\n")
    f.close()

    f = open(fewvalidname, 'w')
    for one in tousevalidres:
        f.write(one + "\n")
    f.close()

    ####test
    f = open(fewtestname, 'w')
    for one in tousetestres:
        f.write(one + "\n")
    f.close()


def getpromptembedding(model, tokenizer, promptnumber, taskname, path):
    t5_embedding = model.model.get_input_embeddings()
    promptinitembedding = torch.FloatTensor(promptnumber, t5_embedding.weight.size(1))
    startindex = 0
    alllabel = ["summarization"]
    alllabel.append(taskname)
    for one in alllabel:
        encoderes = tokenizer.batch_encode_plus([one], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        embeddingres = t5_embedding(touse).clone().detach()
        if embeddingres.shape[0] > 1:
            embeddingres = torch.mean(embeddingres, 0, keepdim=True)
        promptinitembedding[startindex] = embeddingres
        startindex += 1
    fr = open(path, 'rb')
    alltokens = pickle5.load(fr)
    sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
    top5000 = []
    for one in sortedalltoken:
        if one[0] == 2:
            continue
        else:
            if len(top5000) < 5000:
                top5000.append(one)
            else:
                break
    vocab = tokenizer.get_vocab()
    randomtokennum = promptnumber - len(alllabel)
    touse = random.sample(top5000, randomtokennum)
    # print(touse)
    for one in touse:
        promptinitembedding[startindex] = t5_embedding.weight[one[0]].clone().detach()
        startindex += 1
    
    return promptinitembedding


def getmixpromptembedding(model, tokenizer, task_prompt_length, path):
    def sample_top_k_tokens(topk, t5_embedding):
        with open('../support_files/allnumber_t5.pkl', 'rb') as fr:
            alltokens = pickle5.load(fr)
        sortedalltoken = sorted(alltokens.items(), key=lambda item: item[1], reverse=True)
        top5000 = []
        for one in sortedalltoken:
            if one[0] == 2:
                continue
            else:
                if len(top5000) < 5000:
                    top5000.append(one)
                else:
                    break
        vocab = tokenizer.get_vocab()
        while True:
            topk_emb = []
            touse = random.sample(top5000, topk)
            for tok in touse:
                topk_emb.append(t5_embedding.weight[tok[0]].clone().detach().unsqueeze(0))
            yield torch.cat(topk_emb, 0)

    def get_embs(toks, t5_embedding):
        encoderes = tokenizer.batch_encode_plus([toks], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        embeddingres = t5_embedding(touse).clone().detach()
        return embeddingres
    t5_embedding = model.model.get_input_embeddings()
    embeddingres = get_embs("summarize this article:", t5_embedding)
    embs_dict = {}
    embs_dict['__task__'] = next(sample_top_k_tokens(task_prompt_length, t5_embedding))
    embs_dict['__task__'][:embeddingres.size(0)] = embeddingres # set meaningful initial tokens 
    
    return embs_dict


def load_frost(model, args):
    # load tensorflow checkpoint
    ckpt_path = "frost_baseline/frost_ckpt/frost_ckpt"
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

    for k in model.state_dict().keys():
        if not (k in d.keys()):
            #print("Missing from d: {}".format(k), model.state_dict()[k].sum())
            d[k] = model.state_dict()[k]

    for k in d.keys():
        if not (k in model.state_dict().keys()):
            print("In d but not in state_dict: {}".format(k))

    model.load_state_dict(d)
    print("loaded FROST tensorflow weights weights!")

    return model



