# standard fine-tuning

import math
import os
import pdb
import torch
import torch.nn as nn

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig



class ModelFinetune(nn.Module):
    def __init__(self, args, model, tokenizer, model_name):
        super(ModelFinetune, self).__init__()
        self.args = args
        self.model = model
        self.model_name = model_name
        ### load ckpt
        if 'T5' in self.model_name: #only T5 has the option to load lm_adapted
            if args.use_lm_adapted == True:
                print("use lm adapted model!")
                t5ckpt = torch.load(args.lm_adapted_path)
                self.model.load_state_dict(t5ckpt)
        ### for fine-tuning, set requires_grad True
        for name, param in self.model.named_parameters():
            #print(name)
            param.requires_grad = True
        self.tokenizer = tokenizer
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        if args.label_smoothing > 0:
            self.loss_fct = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing)

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        if 'T5' in self.model_name:
            input_embed_part = self.model.encoder.embed_tokens(input_ids)
        elif "Pegasus" in self.model_name or 'Bart' in self.model_name:
            input_embed_part = self.model.get_encoder().embed_tokens(input_ids)
            embed_dim = self.model.config.d_model
            embed_scale = math.sqrt(embed_dim)
            input_embed_part = input_embed_part * embed_scale

        return self.model(
            inputs_embeds=input_embed_part,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        
    def forward(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self._step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        if self.args.label_smoothing > 0:
            logits = outputs[1]
            flat_logits = logits.reshape((logits.shape[0] * logits.shape[1], logits.shape[2]))
            flat_labels = lm_labels.reshape((lm_labels.shape[0] * lm_labels.shape[1]))
            loss = self.loss_fct(flat_logits, flat_labels)

        return loss

    def _generative_step(self, batch):
        if 'T5' in self.model_name:
            input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        elif "Pegasus" in self.model_name or 'Bart' in self.model_name:
            input_embed_part = self.model.get_encoder().embed_tokens(batch["input_ids"])
            embed_dim = self.model.config.d_model
            embed_scale = math.sqrt(embed_dim) 
            input_embed_part = input_embed_part * embed_scale

        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=input_embed_part,
            decoder_input_ids=decoder_input_ids,
            attention_mask=batch["attention_mask"],
            #decoder_attention_mask=batch['target_mask'],
            max_length = self.args.max_summary_length,
            num_beams = self.args.num_beams,
            repetition_penalty = self.args.repetition_penalty,
            length_penalty = self.args.length_penalty,
            use_cache = True,
            early_stopping = True
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
        input = self.ids_to_clean_text(batch["input_ids"])

        return input, target, preds

    def _generative_samples(self, batch):
        if 'T5' in self.model_name:
            input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        elif "Pegasus" in self.model_name or 'Bart' in self.model_name:
            input_embed_part = self.model.get_encoder().embed_tokens(batch["input_ids"])
            embed_dim = self.model.config.d_model
            embed_scale = math.sqrt(embed_dim)
            input_embed_part = input_embed_part * embed_scale

        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0],1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )

        generated_ids = self.model.generate(
            inputs_embeds=input_embed_part,
            decoder_input_ids=decoder_input_ids,
            attention_mask=batch["attention_mask"],
            max_length = self.args.max_length,
            repetition_penalty = self.args.repetition_penalty,
            length_penalty = self.args.length_penalty,
            use_cache = True,
            early_stopping = True,
            do_sample = True,
            top_k = 64,
            #top_p = 0.85,
            num_return_sequences = 4
        )

        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
        input = self.ids_to_clean_text(batch["input_ids"])
        
        return input, target, preds

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        """list(map(f, x))"""
        
        return list(map(f, x))
