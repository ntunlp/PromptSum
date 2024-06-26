import os
import pdb
import sys
import math
import torch
import torch.nn as nn

from torch.nn.functional import kl_div
from torch.nn import Softmax


class ModelSummaryMix(nn.Module):
    def __init__(self, args, model, tokenizer, model_name):
        super(ModelSummaryMix, self).__init__()
        self.args = args
        self.model = model
        self.model_name = model_name
        ### load ckpt
        if 'T5' in self.model_name: #only T5 has the option to load lm_adapted
            if args.use_lm_adapted == 1:
                print("use lm adapted model!")
                t5ckpt = torch.load(args.lm_adapted_path)
                self.model.load_state_dict(t5ckpt)
        for name, param in self.model.named_parameters():
            if args.model == 'BartMixPromptUnfreeze':
                if not("shared" in name):
                    param.requires_grad = False
            else:
                param.requires_grad = False
        self.tokenizer = tokenizer
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        self.promptnumber = 0
        self.promptembedding = None
        self.softmax = Softmax(dim=2)

    def set_prompt_embedding(self, promptnumber, promptembedding):
        self.promptnumber = promptnumber
        self.promptembedding = nn.parameter.Parameter(promptembedding)

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None,
            ent_ids=None, ent_mask=None
    ):
        if 'T5' in self.model_name:
            input_embed_part = self.model.encoder.embed_tokens(input_ids)
        else:
            input_embed_part = self.model.get_encoder().embed_tokens(input_ids)

        # prompt: soft + discrete
        soft_prompt_embed= self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        if 'T5' in self.model_name:
            discrete_prompt_embed = self.model.encoder.embed_tokens(ent_ids)
        else:
            discrete_prompt_embed = self.model.get_encoder().embed_tokens(ent_ids)
        if self.args.use_entity_chain:
            prompt_embed = torch.cat([soft_prompt_embed, discrete_prompt_embed], 1)
        else:
            prompt_embed = soft_prompt_embed
        mask_prompt = torch.full((attention_mask.shape[0], prompt_embed.shape[1]), 1).to(self.args.device)

        if self.args.concat_mode == "concat_right":
            allembedding = torch.cat([input_embed_part, prompt_embed], 1)
            all_attention_mask = torch.cat([attention_mask, mask_prompt], 1)
        else:
            allembedding = torch.cat([prompt_embed, input_embed_part], 1)
            all_attention_mask = torch.cat([mask_prompt, attention_mask], 1)
        if "Pegasus" in self.model_name:
            embed_dim = self.model.config.d_model
            embed_scale = math.sqrt(embed_dim)
            allembedding = allembedding * embed_scale

        return self.model(
            inputs_embeds=allembedding,
            attention_mask=all_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True
        )

    def forward(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self._step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            ent_ids=batch["ents_ids"],
            ent_mask=batch["ents_mask"]
        )
        loss = outputs[0]
        
        return loss

    def _generative_step(self, batch):
        if 'T5' in self.model_name:
            input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        else:
            input_embed_part = self.model.get_encoder().embed_tokens(batch["input_ids"])

        # prompt: soft + discrete
        soft_prompt_embed = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        if 'T5' in self.model_name:
            discrete_prompt_embed = self.model.encoder.embed_tokens(batch["ents_ids"])
        else:
            discrete_prompt_embed = self.model.get_encoder().embed_tokens(batch["ents_ids"])
        if self.args.no_sprompt:
            prompt_embed = discrete_prompt_embed
        elif self.args.no_entity_chain:
            prompt_embed = soft_prompt_embed
        else:
            prompt_embed = torch.cat([soft_prompt_embed, discrete_prompt_embed], 1)
        mask_prompt = torch.full((batch["attention_mask"].shape[0], prompt_embed.shape[1]), 1).to(self.args.device)

        allembedding = torch.cat([input_embed_part, prompt_embed], 1)
        if "Pegasus" in self.model_name:
            embed_dim = self.model.config.d_model
            embed_scale = math.sqrt(embed_dim)
            allembedding = allembedding * embed_scale

        all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )

        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            max_length=self.args.max_summary_length,
            num_beams=self.args.num_beams,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            early_stopping=True
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
