"""
自定义Qwen2模型的forward函数
"""
import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
from typing import Optional, List, Union, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPast
import logging
from typing_extensions import Unpack, TypedDict
from functools import partial
from transformers.utils import logging as hf_logging
from transformers.models.qwen2.modeling_qwen2 import Cache, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import FlashAttentionKwargs

def custom_qwen2_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            #logger.warning_once(
            #    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            #)
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if flash_attn_kwargs.get("cu_kvlen",None) is not None:
            old_position_ids = position_ids.clone()
            cu_kvlen = flash_attn_kwargs["cu_kvlen"] 
            p=0
            for i in range(len(cu_kvlen)-1):
                while(p<len(position_ids[0])):
                    position_ids[0][p]+=cu_kvlen[i+1]-cu_kvlen[i]
                    p+=1
                    if(p==len(position_ids[0]) or position_ids[0][p]==0):
                        break

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_ids = old_position_ids if flash_attn_kwargs.get("cu_kvlen",None) is not None else position_ids
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def apply_custom_qwen2_forward():
    """
    应用自定义的Qwen2 forward函数
    """
    print("🔧 正在替换Qwen2Model的forward函数...")
    
    # 保存原始的forward函数（如果需要恢复的话）
    if not hasattr(Qwen2Model, '_original_forward'):
        Qwen2Model._original_forward = Qwen2Model.forward
    
    # 替换为自定义的forward函数
    Qwen2Model.forward = custom_qwen2_forward
    
    print("✅ Qwen2Model forward函数替换完成!")


def restore_original_qwen2_forward():
    """
    恢复原始的Qwen2 forward函数
    """
    if hasattr(Qwen2Model, '_original_forward'):
        Qwen2Model.forward = Qwen2Model._original_forward
        print("✅ 已恢复原始的Qwen2Model forward函数")
    else:
        print("⚠️ 未找到原始的forward函数备份")


# 自动应用patch
if __name__ == "__main__":
    apply_custom_qwen2_forward()