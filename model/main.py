import copy
from typing import Optional, Tuple, Union
from transformers import T5PreTrainedModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqLMOutput, BaseModelOutputWithPastAndCrossAttentions)
from transformers.models.t5.modeling_t5 import (T5Stack, T5Block, T5LayerNorm)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from model.utils import exact_match
import pdb
from collections import defaultdict


class T5CustomEncoder(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.whole_word_embedding = nn.Embedding(512, config.d_model)
        self.post_init()
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids=None,
            whole_item_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")
        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds + self.whole_word_embedding(whole_item_ids)
        batch_size, seq_length = input_shape
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )
        if past_key_values is None:
            past_key_values = [None] * len(self.block)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout(inputs_embeds)
        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            hidden_states, present_key_value_state = layer_outputs[:2]
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class T5SequentialRecommender(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5CustomEncoder(encoder_config, self.shared)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.h_embed = nn.Embedding(config.vocab_size_prior, config.d_model)
        self.num_indexes = 2
        self.num_templates = 10

        self.encoder_prior = T5CustomEncoder(encoder_config, self.h_embed)
        self.prior = nn.Linear(config.d_model, self.num_indexes * self.num_templates)
        self.prior_I = nn.Linear(config.d_model, self.num_indexes)
        self.prior_T = nn.Linear(config.d_model, self.num_templates)
        
        self.prior_gain_I = nn.Parameter(torch.tensor(1.0))
        self.prior_gain_T = nn.Parameter(torch.tensor(1.0))
        self.prior_gain_IT = nn.Parameter(torch.tensor(1.0))
      
        
        self.model_parallel = False
        self.device_map = None
        self.post_init()

        
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
        # self.encoder_prior.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            whole_item_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                whole_item_ids=whole_item_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        if past is None:
            return past
        reordered_decoder_past = ()
        for layer_past_states in past:
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )
            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def train_step(self, batch, stage=0, k=10, max_len=20, tokenizer = None, constraints=[], use_ll_norm=False, pooling_type='sos', alpha=0, beta=0, tau=1):
        device = next(self.parameters()).device
        # pretraining stage p(y|z)
        if stage == 0:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            whole_item_ids = batch['whole_item_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            target_length = batch['target_length'].to(device)

            I = self.num_indexes
            T = self.num_templates
           
            B, L = target_ids.size()
            y_pred = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            whole_item_ids=whole_item_ids,
            labels=target_ids,
            return_dict=True
            )
            y_pred_logits = y_pred.logits            # b, len, vocab
            
            lm_mask = target_ids != -100
            lm_mask = lm_mask.float()
           
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction="none")
            loss = loss_fct(y_pred_logits.view(-1, y_pred_logits.size(-1)), target_ids.view(-1))
            loss = loss.view(B, L) * lm_mask
            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

            return {
                "total_loss": loss.mean(),
                "ce": loss.mean()
            }               
        # prior stage p(z|h) ~ q(z|h,y)
        elif stage == 1:
            h_input_ids = batch['h']['input_ids'].to(device)
            h_attention_mask = batch['h']['attention_mask'].to(device)
            h_whole_item_ids = batch['h']['whole_item_ids'].to(device)
        
            # z
            z_input_ids = batch['z']['input_ids'].to(device)
            z_attention_mask = batch['z']['attention_mask'].to(device)
            z_whole_item_ids = batch['z']['whole_item_ids'].to(device)
            z_target_ids = batch['z']['target_ids'].to(device)
            z_target_length = batch['z']['target_length'].to(device)

            I = self.num_indexes
            T = self.num_templates
            B, L = h_input_ids.size()
            
            y_pred = self(
                input_ids=z_input_ids,
                attention_mask=z_attention_mask,
                whole_item_ids=z_whole_item_ids,
                labels=z_target_ids,
                return_dict=True
            )
            y_pred_logits = y_pred.logits

            ce = F.cross_entropy(y_pred_logits.transpose(1,2), z_target_ids, ignore_index=-100, reduction='none')         # [B·I·T, Lt]
            mask = (z_target_ids != -100).float()
            ce   = (ce * mask).sum(1) / mask.sum(1).clamp(min=1) # [B·I·T] # nll loss
            log_p_y_given_z = (-ce).view(B, I, T)

            h_out = self.encoder_prior(
                input_ids=h_input_ids,
                attention_mask=h_attention_mask,
                whole_item_ids=h_whole_item_ids,
                return_dict=True
            )
            if pooling_type == 'avg':
                masked_hidden = h_out.last_hidden_state * h_attention_mask.unsqueeze(-1)   # [B, L, D]
                sum_hidden   = masked_hidden.sum(dim=1)                                   # [B, D]
                lengths      = h_attention_mask.sum(dim=1, keepdim=True)                  # [B, 1]
                h_hidden_state  = sum_hidden / lengths.clamp(min=1)    
            elif pooling_type == 'sos':
                h_hidden_state = h_out.last_hidden_state[:, 0, :]
            else:
                raise NotImplementedError(f"Pooling type {pooling_type} not implemented")
                
            
            aaa = self.prior_I(h_hidden_state).unsqueeze(2) * self.prior_gain_I           # (B, I, 1)
            bbb = self.prior_T(h_hidden_state).unsqueeze(1) * self.prior_gain_T         # (B, 1, T)
            base = (aaa + bbb).reshape(B, I*T)   # (B, I*T)
            cross = self.prior(h_hidden_state) * self.prior_gain_IT            # (B, I*T)
            prior_logit = (base + cross) 
            log_prior = F.log_softmax(prior_logit, dim=1)

            log_py = log_p_y_given_z.detach()
            log_pdet = log_prior.view(B, I, T).detach()
            logw = log_py/tau + alpha * log_pdet

            log_q = F.log_softmax(logw.view(B, -1), dim=1).view(B, I * T)
            q_prob = log_q.exp()
            h_q = - q_prob * log_q
            h_q = h_q.sum(1).mean()

            exp_ll = (q_prob * (log_py.view(B,-1))).sum(dim=1).mean()          
            kl = (q_prob * (log_q - log_prior)).sum((1)).mean()
            

            h_prior = - log_prior * log_prior.exp()
            h_prior = h_prior.sum((1)).mean()

            elbo = exp_ll - kl
            
            
            return {
                "total_loss": -elbo - h_prior * beta,
                "kl": kl,
                "h_q": h_q,
                "h_p": h_prior
                # "h_ll": h_ll
            }
      

    @torch.no_grad()
    def get_prior_probs(self, batch, tau=1, pooling_type='sos'):
        self.eval()
        device = next(self.parameters()).device
        I, T = self.num_indexes, self.num_templates   # ex) 2, 10

        h_ids = batch['h']['input_ids'].to(device)
        h_mask = batch['h']['attention_mask'].to(device)
        h_items = batch['h']['whole_item_ids'].to(device)
    
        I = self.num_indexes
        T = self.num_templates
        B, L = h_ids.size()
               
    
        h_out = self.encoder_prior(
            input_ids      = h_ids,
            attention_mask = h_mask,
            whole_item_ids = h_items,
            return_dict=True
        )

        if pooling_type == 'avg':
            masked_hidden = h_out.last_hidden_state * h_mask.unsqueeze(-1)   # [B, L, D]
            sum_hidden   = masked_hidden.sum(dim=1)                                   # [B, D]
            lengths      = h_mask.sum(dim=1, keepdim=True)                  # [B, 1]
            h_hidden_state  = sum_hidden / lengths.clamp(min=1)    
        elif pooling_type == 'sos':
            h_hidden_state = h_out.last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError(f"Pooling type {pooling_type} not implemented")
        
        aaa = self.prior_I(h_hidden_state).unsqueeze(2) * self.prior_gain_I           # (B, I, 1)
        bbb = self.prior_T(h_hidden_state).unsqueeze(1) * self.prior_gain_T         # (B, 1, T)
        base = (aaa + bbb).reshape(B, I*T)   # (B, I*T)
        cross = self.prior(h_hidden_state) * self.prior_gain_IT            # (B, I*T)
        prior_logit = (base + cross) 
        log_prior = F.log_softmax(prior_logit, dim=1)
        prior_prob = log_prior.exp()
        # log_prior = F.log_softmax(prior/tau, dim=1).view(B, I, T)
        return prior_prob, prior_logit

    @torch.no_grad()
    def generate_step(self, batch, k=10, max_len=20, constraint=None, chosen_idx=None):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['z']['input_ids'].to(device)
        whole_item_ids = batch['z']['whole_item_ids'].to(device)
        if chosen_idx is not None:
            input_ids = input_ids[chosen_idx]
            whole_item_ids = whole_item_ids[chosen_idx]
        # lm_labels = batch["target_ids"].to(device)
        attention_mask = input_ids.ne(0).to(dtype=torch.float32, device=device)
        beam_out = self.generate(
            input_ids=input_ids,
            whole_item_ids=whole_item_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            prefix_allowed_tokens_fn=constraint,
            num_beams=k,
            num_return_sequences=k,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
            early_stopping=True)
        return beam_out

    def valid_pretrain(self, batch, k=10, max_len=20, constraints=[], tokenizer=None):
        self.eval()

        device = next(self.parameters()).device
        B, _ = batch['h']['input_ids'].size()
        I, T = self.num_indexes, self.num_templates
        
        
        z_ids   = batch['z']['input_ids'     ].to(device)
        z_mask  = batch['z']['attention_mask'].to(device)
        z_items = batch['z']['whole_item_ids'].to(device)
        z_labels = batch['z']["target_ids"].to(device)
        _, L = z_labels.size()

        offset  = torch.arange(B, device=device) * (I * T)          # [B]

        idx_0_0 = offset + 0*T + 0
        idx_1_0 = offset + 1*T + 0

        hit_1, hit_5, hit_10, ndcg_5, ndcg_10, i_pred, t_pred = 0, 0, 0, 0, 0, 0, 0

        hit_1_ceid, hit_5_ceid, hit_10_ceid, ndcg_5_ceid, ndcg_10_ceid, corr_list_ceid = 0, 0, 0, 0, 0, []
        hit_1_seid, hit_5_seid, hit_10_seid, ndcg_5_seid, ndcg_10_seid, corr_list_seid = 0, 0, 0, 0, 0, []


        for index_seq, (idx, constraint) in enumerate(zip([idx_0_0, idx_1_0], constraints)):
            idx = idx.to(device)
            z_labels = batch['z']["target_ids"].to(device)[idx]
            beam_out = self.generate_step(batch, k=k, max_len=max_len, constraint=constraint, chosen_idx=idx)
            lm_labels = torch.where(z_labels == -100, 0, z_labels)
            gold_sents = tokenizer.batch_decode(lm_labels, skip_special_tokens=True)
            generated_sents = tokenizer.batch_decode(beam_out['sequences'], skip_special_tokens=True)
            one_hit_1, one_hit_5, one_hit_10, one_ndcg_5, one_ndcg_10, corr_list = exact_match(generated_sents, beam_out['sequences_scores'], gold_sents, k)
            if index_seq == 0:
                hit_1_ceid += one_hit_1
                hit_5_ceid += one_hit_5
                hit_10_ceid += one_hit_10
                ndcg_5_ceid += one_ndcg_5
                ndcg_10_ceid += one_ndcg_10
            else:
                hit_1_seid += one_hit_1
                hit_5_seid += one_hit_5
                hit_10_seid += one_hit_10
                ndcg_5_seid += one_ndcg_5
                ndcg_10_seid += one_ndcg_10
        return hit_1, hit_5, hit_10, ndcg_5, ndcg_10, i_pred, t_pred, hit_1_ceid, hit_5_ceid, hit_10_ceid, ndcg_5_ceid, ndcg_10_ceid, hit_1_seid, hit_5_seid, hit_10_seid, ndcg_5_seid, ndcg_10_seid
    
 

    @torch.no_grad()
    def inference_all(self, batch, k=10, max_len=20, constraints=[], tokenizer=None, idx_iid={}, index_list=[0,1], template_num=10, tau=1, pooling_type='sos'):
        self.eval()
        device = next(self.parameters()).device
        
        prior, prior_logits = self.get_prior_probs(batch, tau, pooling_type)
        B, I, T = prior.size(0), self.num_indexes, self.num_templates
        T_sel = template_num
        # template_set = range(T_sel)
        index_set = set(index_list)

        base = torch.arange(B, device=device)*(I*T)
        templ = torch.arange(T_sel, device=device)

        z_ids   = batch['z']['input_ids'     ].to(device)
        z_mask  = batch['z']['attention_mask'].to(device)
        z_items = batch['z']['whole_item_ids'].to(device)
        z_labels = batch['z']["target_ids"].to(device)
        lm_labels = torch.where(z_labels == -100, 0, z_labels)

        results = {'ceid': {}, 'seid': {}}

        y_pred = self(
            input_ids=z_ids,
            attention_mask=z_mask,
            whole_item_ids=z_items,
            labels=z_labels,
            return_dict=True
            )
        y_pred_logits = y_pred.logits            # b, len, vocab

        ce = F.cross_entropy(y_pred_logits.transpose(1,2), z_labels, ignore_index=-100, reduction='none')         # [B·I·T, Lt]
        mask = (z_labels != -100).float()
        ce   = (ce * mask).sum(1) / mask.sum(1).clamp(min=1) # [B·I·T]    
        log_p_y_given_z = (-ce).view(B, I, T)

        for index_idx in index_set:         
            out_dict = {
                'gt': [], # ['user_idx'] # b
                'iid': [],  # ['user_idx'] # b
                'preds' : [], # ['user_idx', template_idx, k] # b, T_sel, k
                'preds_iid':[], # ['user_idx', template_idx, k] # b, T_sel, k
                'prior_prob' : [], # ['user_idx', template_idx] # b, T_sel
                "prior_logit": [],
                'log_prob' : [], # ['user_idx', template_idx, k] # b, T_sel, k
                "log_ll": [], # ['user_idx', template_idx] # b, T_sel
                'ranks' : [], # ['user_idx', template_idx] # b, T_sel
            }

            constraint_fn = constraints[index_idx]
            base = base + index_idx*T
            view_rows = base[:, None] + templ[None, :]
            view_rows = view_rows.view(-1)
            
            prediction = self.generate_step(batch, k=k, constraint=constraint_fn, chosen_idx= view_rows)        
            generated_sents = tokenizer.batch_decode(prediction['sequences'], skip_special_tokens=True)
            gold_sents = tokenizer.batch_decode(lm_labels[base], skip_special_tokens=True)
            gts = gold_sents
            iids = [idx_iid[gt] for gt in gts]

            preds = [generated_sents[b*T_sel*k:(b+1)*T_sel*k] for b in range(B)]
            preds = [[pred[t*k:(t+1)*k] for t in range(T_sel)] for pred in preds]

            generated_iids = [idx_iid[pred] for pred in generated_sents]
            generated_iids = [generated_iids[b*T_sel*k:(b+1)*T_sel*k] for b in range(B)]
            generated_iids = [[generated_iids[t*k:(t+1)*k] for t in range(T_sel)] for generated_iids in generated_iids]

            prior_prob = prior.view(B, I, T_sel)[:, index_idx, :].tolist()
            prior_logit = prior_logits.view(B, I, T_sel)[:, index_idx, :].tolist()
            log_prob = prediction['sequences_scores'].view(B, T_sel, k).tolist()


            ranks = [[] for _ in range(B)]
            for b in range(B):
                for t in range(T_sel):
                    try:
                        rank = generated_iids[b][t].index(iids[b]) + 1
                    except ValueError:
                        rank = -1
                    ranks[b].append(rank)      

            log_ll = log_p_y_given_z[:, index_idx, :].tolist()          
            out_dict['gt'] = gts
            out_dict['iid'] = iids
            out_dict['preds'] = preds
            out_dict['preds_iid'] = generated_iids
            out_dict['prior_prob'] = prior_prob
            out_dict['prior_logit'] = prior_logit
            out_dict['log_prob'] = log_prob
            out_dict['log_ll'] = log_ll
            out_dict['ranks'] = ranks
            

            if index_idx == 0:
                results['ceid'] = out_dict
                
            else:
                results['seid'] = out_dict

        return results
         