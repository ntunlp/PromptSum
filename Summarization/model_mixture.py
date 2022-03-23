# Prompt tuning mixed with entities with T5

import os
import pdb
import torch
import torch.nn as nn
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config



class T5MixPrompt(nn.Module):
    def __init__(self, args, model, tokenizer):
        super(T5MixPrompt, self).__init__()
        self.args = args
        self.model = model
        ### load ckpt
        if args.use_lm_adapted == True:
            print("use lm adapted model!")
            t5ckpt = torch.load(args.lm_adapted_path)
        ### if prompt tuning, set requires_grad false
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        self.tokenizer = tokenizer
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        self.promptnumber = 0
        self.prompt_dict = nn.ParameterDict() # {label_name: [label_name_emb, label_soft_tokens]}, specially, self.prompt_dict['__task__']: task_soft_tokens
        self.prompt_fix_dict = {}
        self.mode = args.concat_mode
        self.seen_labels_cl = set() # seen labels so far, under continual learning

    def add_seen_labels(self, labels):
        self.seen_labels_cl.update(labels)

    def set_prompt_embedding(self, promptnumber, task_emb):
        self.promptnumber = promptnumber
        self.prompt_dict['__task__'] = nn.parameter.Parameter(task_emb['__task__'].to(self.model.device))

    def _construct_prompt_batch(self, batchsize, ent_ids):
        prompt_embs = []
        for idx in range(batchsize):
            prompt_embs.append(self._construct_prompt(ent_ids[idx]).unsqueeze(0))
        
        return torch.cat(prompt_embs, 0)
    
    def _construct_prompt(self, ent_ids):
        prompt_emb = []
        # append task soft prompt 
        prompt_emb.append(self.prompt_dict['__task__'])
        # append ent fixed prompt
        if ent_ids.nelement() > 0: # possibly encounter empty entity guidance
            prompt_emb += [self.model.encoder.embed_tokens(ent_ids)]
        
        return torch.cat(prompt_emb, 0)

    def _step(
            self, input_ids, ent_ids, attention_mask=None, ent_attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None, labels_set=None
    ):
        ##### handle prompt, cal input_embed
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        
        prompt_embedding = self._construct_prompt_batch(batchsize=input_embed_part.size(0), ent_ids=ent_ids)
        prompt_length = prompt_embedding.size(1)
        if ent_attention_mask is None:
            mask_prompt = torch.full((attention_mask.shape[0], prompt_length),1).to(self.args.device)
        else:
            mask_prompt = torch.cat([torch.full((attention_mask.shape[0], self.promptnumber),1).to(self.args.device), ent_attention_mask], 1)

        if self.mode == 'right_concat':
            allembedding = torch.cat([input_embed_part, prompt_embedding], 1)
            all_attention_mask = torch.cat([attention_mask, mask_prompt], 1)
        if self.mode == 'left_concat':
            allembedding = torch.cat([prompt_embedding, input_embed_part], 1)
            all_attention_mask = torch.cat([mask_prompt, attention_mask], 1)
        
        return self.model(
            inputs_embeds=allembedding,
            attention_mask=all_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def forward(self, batch, labels_set=None):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self._step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            ent_attention_mask=batch["ents_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            ent_ids=batch['input_ents'],
            labels_set=labels_set,
        )
        loss = outputs[0]
        
        return loss

    def _generative_step(self, batch):
        input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        prompt_embedding = self._construct_prompt_batch(batchsize=input_embed_part.size(0), ent_ids=batch['input_ents'])
        if self.mode == 'right_concat':
            allembedding = torch.cat([input_embed_part, prompt_embedding], 1)
        elif self.mode == 'left_concat':
            allembedding = torch.cat([prompt_embedding, input_embed_part], 1)
        prompt_length = prompt_embedding.size(1)
        if 'ents_mask' not in batch:
            mask_prompt = torch.full((batch["attention_mask"].shape[0], prompt_length), 1).to(self.args.device)
        else:
            mask_prompt = torch.cat([torch.full((batch["attention_mask"].shape[0], self.promptnumber),1).to(self.args.device), batch['ents_mask']], 1)
        if self.mode == 'right_concat':
            all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        elif self.mode == 'left_concat':
            all_attention_mask = torch.cat([mask_prompt, batch["attention_mask"]], 1)
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            #decoder_attention_mask=batch['target_mask'],
            max_length=128,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
        input = self.ids_to_clean_text(batch["input_ids"])
        ents = self.ids_to_clean_text(batch["input_ents"])
        
        return input, target, preds

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        """list(map(f, x))"""
        
        return list(map(f, x))
