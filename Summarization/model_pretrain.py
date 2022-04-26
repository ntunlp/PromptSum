import os
import pdb
import torch
import torch.nn as nn
import gc

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config



class T5forPretrain(nn.Module):
    def __init__(self, args, model, tokenizer):
        super(T5forPretrain, self).__init__()
        self.args = args
        self.model = model
        ### load ckpt
        t5ckpt = torch.load(args.lm_adapted_path)
        self.model.load_state_dict(t5ckpt)
        if not (args.pretrain and args.pretrain_all_weights):
            for name, param in self.model.named_parameters():
                param.requires_grad = False
        self.tokenizer = tokenizer
        self.decoder_start_token_id_use = self.model.config.decoder_start_token_id
        self.promptnumber = 0
        self.promptembedding = None
        self.promptnumberforsum = 0
        self.promptembeddingforsum = None

    def set_prompt_embedding(self,promptnumber,promptembedding):
        self.promptnumber = promptnumber
        self.promptembedding = nn.parameter.Parameter(promptembedding)

    def set_prompt_embedding_sum(self,promptnumber,promptembedding):
        self.promptnumberforsum = promptnumber
        self.promptembeddingforsum = nn.parameter.Parameter(promptembedding)

    def _step(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
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

    def _step_sum(
            self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None
    ):
        input_embed_part = self.model.encoder.embed_tokens(input_ids)
        prompt_embed_repeat = self.promptembeddingforsum.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        mask_prompt = torch.full((attention_mask.shape[0], self.promptnumberforsum), 1).to(self.args.device)
        all_attention_mask = torch.cat([attention_mask, mask_prompt], 1)

        return self.model(
            inputs_embeds=allembedding,
            attention_mask=all_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

    def forward(self, batch):
        lm_labels = batch["entity_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self._step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['entity_mask']
        )
        lossent = outputs[0]

        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self._step_sum(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        losssum = outputs[0]

        return lossent,losssum

    def _generative_step(self, batch):
        input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
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
            max_length=64,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = self.ids_to_clean_text(generated_ids)
        target = self.ids_to_clean_text(batch["entity_ids"])
        input = self.ids_to_clean_text(batch["input_ids"])

        input_embed_part = self.model.encoder.embed_tokens(batch["input_ids"])
        prompt_embed_repeat = self.promptembeddingforsum.repeat(input_embed_part.size(0), 1, 1)
        allembedding = torch.cat([input_embed_part, prompt_embed_repeat], 1)
        mask_prompt = torch.full((batch["attention_mask"].shape[0], self.promptnumberforsum), 1).to(self.args.device)
        all_attention_mask = torch.cat([batch["attention_mask"], mask_prompt], 1)
        decoder_input_ids = (
                torch.ones((batch["input_ids"].shape[0], 1), dtype=torch.long, device=batch["input_ids"].device) * self.decoder_start_token_id_use
        )
        generated_ids = self.model.generate(
            inputs_embeds=allembedding,
            decoder_input_ids=decoder_input_ids,
            attention_mask=all_attention_mask,
            use_cache=True,
            max_length=64,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        predssum = self.ids_to_clean_text(generated_ids)
        targetsum = self.ids_to_clean_text(batch["target_ids"])
        inputsum = self.ids_to_clean_text(batch["input_ids"])

        return inputsum, targetsum, predssum, input, target, preds

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return self.lmap(str.strip, gen_text)

    def lmap(self, f, x):
        """list(map(f, x))"""

        return list(map(f, x))
