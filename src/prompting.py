import time
import random 
import logging
import torch
import pickle5 as pickle



def get_prompt_embedding(model, tokenizer, prompt_length):
    t5_embedding = model.model.get_input_embeddings()
    promptinitembedding = torch.FloatTensor(prompt_length, t5_embedding.weight.size(1))
    startindex = 0
    alllabel = ["summarize this article:"]
    for one in alllabel:
        encoderes = tokenizer.batch_encode_plus([one], padding=False, truncation=False, return_tensors="pt")
        touse = encoderes["input_ids"].squeeze()[:-1]
        embeddingres = t5_embedding(touse).clone().detach()
        if embeddingres.shape[0] > 1:
            embeddingres = torch.mean(embeddingres, 0, keepdim=True)
        promptinitembedding[startindex] = embeddingres
        startindex += 1
    fr = open('allnumber.pickle', 'rb')
    alltokens = pickle.load(fr)
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
    randomtokennum = prompt_length - len(alllabel)
    touse = random.sample(top5000, randomtokennum)
    for one in touse:
        promptinitembedding[startindex] = t5_embedding.weight[one[0]].clone().detach()
        startindex += 1

    return promptinitembedding


def get_mix_prompt_embedding(model, tokenizer):
    def sample_top_k_tokens(topk, t5_embedding):
        with open('allnumber.pickle', 'rb') as fr:
            alltokens = pickle.load(fr)
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


def load_prompt(args, model):
    allckpt = torch.load(args.ckpt_path)
    print(allckpt.keys())
    if args.model == 'T5Prompt':
        model.prompt_length = allckpt["prompt_length"]
        model.prompt_embedding = allckpt["prompt_embedding"]
    elif args.model == 'T5MixPrompt':
        model.prompt_dict = allckpt['prompt_dict']
        model.prompt_fix_dict = allckpt['prompt_fix_dict']
        for k, v in model.prompt_fix_dict.items():
            model.prompt_fix_dict[k] = v.to(args.device)
    elif args.model == 'T5Finetune':
        model.model.load_state_dict(allckpt['t5-base'])



