from __future__ import annotations

import json

from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger

@ModelBase.register("Zamba2ForCausalLM")
@ModelBase.example("Zyphra/Zamba2-1.2B")
class Zamba2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.ZAMBA2

    def __init__(self, dir_model: Path, *args, **kwargs):
        # Avoid using AutoConfig for hparams
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            with open(dir_model / "config.json", "r", encoding="utf-8") as f:
                hparams = json.load(f)
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

    def set_vocab(self):
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        elif (self.dir_model / "tokenizer.json").is_file():
            try:
                self._set_vocab_llama_hf()
            except (FileNotFoundError, TypeError):
                self._set_vocab_gpt2()
        else:
            self._set_vocab_builtin("llama-spm", self.hparams.get("vocab_size", 32000))
    
    
    def set_gguf_parameters(self):
        d_model  = self.find_hparam(["hidden_size", "d_model"])
        d_conv   = self.find_hparam(["mamba_d_conv", "conv_kernel", "d_conv"], optional=True) or 4
        d_state  = self.find_hparam(["mamba_d_state", "state_size", "d_state"], optional=True) or 128
        expand   = self.find_hparam(["mamba_expand"], optional=True) or 2
        d_inner  = expand * d_model
        head_dim = self.find_hparam(["mamba_headdim", "mamba_d_head"], optional=True) or 64
        n_group  = self.find_hparam(["mamba_ngroups", "n_groups"], optional=True) or 1

        rms_norm_eps = self.find_hparam(["rms_norm_eps", "layer_norm_epsilon"], optional=True) or 1e-5

        n_head    = self.find_hparam(["num_attention_heads"])
        n_head_kv = self.find_hparam(["num_key_value_heads"], optional=True) or n_head
        attn_head_dim = self.find_hparam(["attention_head_dim"], optional=True) or (d_model // n_head)

        max_seq_len = self.find_hparam(["max_position_embeddings"], optional=True) or 4096
        ffn_length = self.find_hparam(["ffn_hidden_size", "intermediate_size"], optional=True) or 4 * d_model

        layers_block_type = self.hparams.get("layers_block_type", [])
        n_kv_vec = [
            n_head_kv if layer_type == "hybrid" else 0
            for layer_type in layers_block_type
        ]

        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(max_seq_len)
        self.gguf_writer.add_embedding_length(d_model)
        self.gguf_writer.add_feed_forward_length(ffn_length)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_kv_vec)
        self.gguf_writer.add_key_length(attn_head_dim)
        self.gguf_writer.add_value_length(attn_head_dim)
        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(d_inner // head_dim)
        self.gguf_writer.add_ssm_group_count(n_group)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_rope_freq_base(self.find_hparam(["rope_theta"], optional=True) or 10000.0)
        self.gguf_writer.add_file_type(self.ftype)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        pass

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        pass