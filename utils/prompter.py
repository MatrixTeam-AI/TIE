import html
import string

import ftfy
import regex as re
import torch
from transformers import AutoTokenizer


def basic_clean(text):
    return html.unescape(html.unescape(ftfy.fix_text(text))).strip()


def whitespace_clean(text):
    return re.sub(r"\s+", " ", text).strip()


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text.lower()).strip()


class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, "whitespace", "lower", "canonicalize")
        self.seq_len = seq_len
        self.clean = clean
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)
        tokenizer_kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            tokenizer_kwargs.update({"padding": "max_length", "truncation": True, "max_length": self.seq_len})
        tokenizer_kwargs.update(**kwargs)
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(text) for text in sequence]
        ids = self.tokenizer(sequence, **tokenizer_kwargs)
        if return_mask:
            return ids.input_ids, ids.attention_mask
        return ids.input_ids

    def _clean(self, text):
        if self.clean == "whitespace":
            return whitespace_clean(basic_clean(text))
        if self.clean == "lower":
            return whitespace_clean(basic_clean(text)).lower()
        if self.clean == "canonicalize":
            return canonicalize(basic_clean(text))
        return text


class WanEventPrompter:
    def __init__(self, tokenizer_path=None, text_len=512):
        self.text_len = text_len
        self.text_encoder = None
        self.fetch_tokenizer(tokenizer_path)

    def fetch_tokenizer(self, tokenizer_path=None):
        if tokenizer_path is not None:
            self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=self.text_len, clean="whitespace")

    def fetch_models(self, text_encoder=None):
        self.text_encoder = text_encoder

    def process_prompt(self, prompt, positive=True):
        if positive:
            return prompt
        return prompt or ""

    def encode_event_prompt(self, prompt, positive=True, split_timeline=False, fuse_global_timeline=False, global_timeline_idx=0, device="cuda"):
        texts = prompt[2]
        encoded = [self.tokenizer(text, return_mask=True, add_special_tokens=True) for text in texts]
        seq_lens = [mask.gt(0).sum(dim=1).long() - 1 for (_ids, mask) in encoded]
        seq_lens[-1] = seq_lens[-1] + 1
        ids = torch.cat([ids[:, : seq_lens[i]] for i, (ids, _mask) in enumerate(encoded)], dim=-1)
        total_len = ids.shape[1]
        indices = torch.zeros(ids.shape, dtype=torch.long)
        left = 0
        for i, (_ids, _mask) in enumerate(encoded):
            indices[:, left : left + seq_lens[i]] = i
            left += seq_lens[i]
        if total_len >= self.text_len:
            ids = ids[:, : self.text_len]
            ids[:, -1] = 1
            mask = torch.ones(ids.shape, dtype=torch.int64)
            total_len = self.text_len
            indices = indices[:, : self.text_len]
        else:
            ids = torch.cat([ids, torch.zeros(1, self.text_len - total_len, dtype=torch.long)], dim=-1)
            mask = torch.zeros(ids.shape, dtype=torch.int64)
            mask[:, :total_len] = 1
            indices = torch.cat([indices, -torch.ones(1, self.text_len - total_len, dtype=torch.long)], dim=-1)

        ids = ids.to(device)
        mask = mask.to(device)
        indices = indices.to(device)
        start = torch.tensor([prompt[0]], dtype=torch.float32, device=device)
        end = torch.tensor([prompt[1]], dtype=torch.float32, device=device)
        encoder_dtype = next(self.text_encoder.parameters()).dtype
        if split_timeline and len(prompt) > 3:
            timeline_indexes = prompt[3]
            prompt_emb = torch.zeros((1, self.text_len, self.text_encoder.dim), dtype=encoder_dtype, device=device)
            timelines = list(set(timeline_indexes))
            for timeline in timelines:
                if fuse_global_timeline:
                    if timeline == global_timeline_idx:
                        continue
                    target_prompt_idx = torch.tensor([i for i in range(len(timeline_indexes)) if timeline_indexes[i] in (timeline, global_timeline_idx)], dtype=torch.long, device=device)
                else:
                    target_prompt_idx = torch.tensor([i for i in range(len(timeline_indexes)) if timeline_indexes[i] == timeline], dtype=torch.long, device=device)
                target_id_idx = torch.isin(indices, target_prompt_idx)
                target_id_idx[:, total_len - 1] = True
                prompt_emb[target_id_idx] = self.text_encoder(ids[target_id_idx][None, :], mask[target_id_idx][None, :])
            if fuse_global_timeline:
                target_prompt_idx = torch.tensor([i for i in range(len(timeline_indexes)) if timeline_indexes[i] == global_timeline_idx], dtype=torch.long, device=device)
                target_id_idx = torch.isin(indices, target_prompt_idx)
                target_id_idx[:, total_len - 1] = True
                prompt_emb[target_id_idx] = self.text_encoder(ids[target_id_idx][None, :], mask[target_id_idx][None, :])
        else:
            prompt_emb = self.text_encoder(ids, mask)
            prompt_emb[:, total_len:] = 0
        return prompt_emb, indices, start, end, total_len

    def encode_prompt(self, prompt, positive=True, device="cuda"):
        prompt = self.process_prompt(prompt, positive=positive)
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for seq_len in seq_lens:
            prompt_emb[:, seq_len:] = 0
        return prompt_emb

