import os
import pdb
import torch
import torch.nn as nn
import gc

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config



class ModelforFinetuneEntity(nn.Module):
    def __init__(self, model, tokenizer, args):
        super(ModelforFinetuneEntity, self).__init__()
        self.args = args
        self.model = model
        if 't5' in args.model_name:
            ### load ckpt
            ckpt = torch.load(args.lm_adapted_path)
            self.model.load_state_dict(ckpt)
        elif 'pegasus' in args.model_name:
            ### load ckpt
            ckpt = torch.load(args.pretrain_ckpt, map_location="cuda:0")
            # many times it has module.model. as starting, which is extra
            newdict = {}
            for key in list(ckpt.keys()):
                if key.startswith('module.model.'):
                    newkey = key.replace('module.model.', '')
                    newdict[newkey] = ckpt[key]
                # else:
                #     newdict[key] = t5ckpt[key] #this is "module.promptembedding", "module.promptembeddingforsum", not needed
            self.model.load_state_dict(newdict)
        if not (args.pretrain and args.pretrain_all_weights):
            for name, param in self.model.named_parameters():
                param.requires_grad = False
        self.tokenizer = tokenizer
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        self.promptnumber = 0
        self.promptembedding = None

    def set_prompt_embedding(self, promptnumber, promptembedding):
        self.promptnumber = promptnumber
        self.promptembedding = nn.parameter.Parameter(promptembedding)

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        if 't5' in self.args.model_name:
            encoder = self.model.encoder
        elif 'pegasus' in self.args.model_name:
            encoder = self.model.get_encoder()
        input_embed_part = encoder.embed_tokens(input_ids)
        prompt_embed_repeat = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        mask_prompt = torch.full((attention_mask.shape[0], self.promptnumber),1).to(self.args.device)
        all_attention_mask = torch.cat([attention_mask, mask_prompt], 1)

        return self.model(
            inputs_embeds=allembedding,
            attention_mask=all_attention_mask,
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

        return loss

    def _generative_step_for_tagger(self, batch):
        if 't5' in self.args.model_name:
            encoder = self.model.encoder
        elif 'pegasus' in self.args.model_name:
            encoder = self.model.get_encoder()
        input_embed_part = encoder.embed_tokens(batch["input_ids"])
        soft_prompt_embed = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, soft_prompt_embed], 1)
        mask_prompt = torch.full((batch["attention_mask"].shape[0], self.promptnumber), 1).to(self.args.device)
        all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            max_length=128,
            num_beams=self.args.num_beams,
            repetition_penalty=self.args.repetition_penalty,
            length_penalty=self.args.length_penalty,
            early_stopping=True
        )

        preds = self.ids_to_clean_text(generated_ids)

        return preds

    def _generative_step(self, batch):
        if 't5' in self.args.model_name:
            encoder = self.model.encoder
        elif 'pegasus' in self.args.model_name:
            encoder = self.model.get_encoder()
        input_embed_part = encoder.embed_tokens(batch["input_ids"])
        prompt_embed_repeat = self.promptembedding.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        mask_prompt = torch.full((batch["attention_mask"].shape[0], self.promptnumber), 1).to(self.args.device)
        all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        decoder_input_ids = (
            torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            max_length=128,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["target_ids"])
        input = self.ids_to_clean_text(batch["input_ids"])

        return input,target,preds

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        """list(map(f, x))"""

        return list(map(f, x))
