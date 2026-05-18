import copy

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict, Any
import inspect

from transformers import  PretrainedConfig, AutoConfig, GenerationConfig
from transformers import GenerationMixin
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers.modeling_outputs import Seq2SeqLMOutput,  ModelOutput, BaseModelOutput
from transformers.utils import is_accelerate_available
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .base_model import QwenPreTrainedModel
from .encoder import QwenModelEncoder
from .combine_encoder import QwenModelCombineEncoder
from .decoder import QwenCrossAttDecoder, NLLBDecoder
from .modules.connector import Connector, GroupedEncoderFusion

def print_train_module(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.cuda.current_device() == 0:
                print(f"train  ==> {name}")
class QwenForSeq2SeqConfig(Qwen2Config):
    def __init__(
        self, 
        mt_model_path = None, 
        llm_path = None,
        num_connector_layers= None,
        connector_hidden_size = None,
        connector_intermediate_size = None,
        connector_num_attention_heads = None,
        connector_num_key_value_heads = None,
        connector_model_method = None,
        fuse_model_group_size = None,
        **kwargs):
        super().__init__(**kwargs)
        self.mt_model_path = mt_model_path
        self.llm_path = llm_path
        self.num_connector_layers = num_connector_layers
        self.connector_hidden_size = connector_hidden_size
        self.connector_intermediate_size = connector_intermediate_size
        self.connector_num_attention_heads = connector_num_attention_heads
        self.connector_num_key_value_heads = connector_num_key_value_heads
        self.connector_model_method = connector_model_method
        self.fuse_model_group_size = fuse_model_group_size

class QwenModelForSeq2Seq(QwenPreTrainedModel):
    def __init__(self, config: QwenForSeq2SeqConfig):
        super().__init__(config)

        self.llm_config = AutoConfig.from_pretrained(config.llm_path)
        self.mt_config = AutoConfig.from_pretrained(config.mt_model_path)

        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_path)

        adapter_config = copy.deepcopy(config.to_dict())
        adapter_config["num_hidden_layers"] = config.num_connector_layers
        adapter_config["hidden_size"] = config.connector_hidden_size
        adapter_config["intermediate_size"] = config.connector_intermediate_size
        adapter_config["num_attention_heads"] = config.connector_num_attention_heads
        adapter_config["num_key_value_heads"] = config.connector_num_key_value_heads
        adapter_config["encoder_method"] = config.connector_model_method
        adapter_config["layer_types"] = ["full_attention"] * config.num_connector_layers
        adapter_config["llm_hidden_size"] = self.llm_config.hidden_size
        adapter_config = Qwen2Config(**adapter_config)

        self.adapter_config = adapter_config
        
        self.connector = Connector(self.adapter_config)
        self.fuse_model = GroupedEncoderFusion(self.llm_config, config.fuse_model_group_size)
        self.mt_model = AutoModelForSeq2SeqLM.from_pretrained(config.mt_model_path)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        # if inputs_embeds is None:
        #     inputs_embeds = self.llm.embed_tokens(input_ids)
        # hidden_states = inputs_embeds
        # all_hidden_states = ()
        if labels is not None:
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            output_hidden_states = llm_outputs.hidden_states
            fuse_hidden_state = self.fuse_model(output_hidden_states[1:])  # exclude embedding
            last_hidden_state = fuse_hidden_state

            # last_hidden_state = encoder_outputs.last_hidden_state
            
            connector_outputs = self.connector(
                last_hidden_state,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_states = connector_outputs.last_hidden_state

            decoder_outputs = self.mt_model.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                encoder_hidden_states=last_hidden_states,
                encoder_attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                    # position_ids=position_ids,
            )
                # print(decoder_outputs.keys())
            hidden_states = decoder_outputs[0]
            ## compute loss 
            pretraining_tp = getattr(self.config, "pretraining_tp", 1)
            if pretraining_tp is not None and pretraining_tp > 1:
                print(f"Using pretraining_tp={pretraining_tp} for parallel LM head computation!")
                lm_head_slices = self.mt_model.lm_head.weight.split(self.vocab_size // pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.mt_model.lm_head(hidden_states)
            # print(f"Logits shape: {logits.shape}, labels shape: {labels.shape}")
            logits = logits.float()
            
            loss = None
            if labels is not None:
                # print(f"labels shape: {labels.shape}, logits shape: {logits.shape}")
                loss_fct = CrossEntropyLoss()
                labels = labels.to(logits.device)
                lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = lm_loss
                if self.contrastive_lambda > 0:
                    contrastive_loss = self.compute_contrastive_loss(last_hidden_states, hidden_states)
                    # contrastive_loss = self.compute_contrastive_loss(encoder_all_hidden_states[-1], dec_hidden_states)
                    loss = lm_loss + self.contrastive_lambda * contrastive_loss
                if self.ot_lambda > 0:
                    ot_loss = self.compute_ot_loss_cosine(
                        encoder_hidden_states=last_hidden_states,
                        mask_a=attention_mask,
                        hidden_states_b=hidden_states,
                        mask_b=decoder_attention_mask,
                        reg=self.ot_reg,
                        num_iters=self.ot_num_iters,
                        eps=self.ot_eps
                    )
                    print(f"OT Loss: {ot_loss.item():.4f}")
                    loss = loss + self.ot_lambda * ot_loss

            if not return_dict:
                    output = (logits,) + decoder_outputs[1:]
                    return (loss,) + output if loss is not None else output

            return Seq2SeqLMOutput(
                loss=loss,
                logits=logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                # decoder_hidden_states=decoder_outputs,
                cross_attentions=None,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        else:
            
            llm_outputs = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            output_hidden_states = llm_outputs.hidden_states
            fuse_hidden_state = self.fuse_model(output_hidden_states[1:])  # exclude embedding
            last_hidden_state = fuse_hidden_state

            # last_hidden_state = encoder_outputs.last_hidden_state
            
            connector_outputs = self.connector(
                last_hidden_state,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            last_hidden_states = connector_outputs.last_hidden_state
            
            encoder_outputs = BaseModelOutput(
                last_hidden_state=last_hidden_states,
            )
            
            
            # decoder_generate_ids_list = []

            generation_config = GenerationConfig(
                forced_bos_token_id=decoder_input_ids[0, 0].item(),
                max_new_tokens=200,
                num_beams=5
            )
            

            decoder_generate_ids = self.mt_model.generate(
                input_ids=decoder_input_ids,
                # forced_bos_token_id=decoder_input_ids[0, 0].item(),
                generation_config=generation_config,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask
            )

            # decoder_generate_ids_list.append(decoder_generate_ids)

            if len(decoder_generate_ids) == 1:
                return (input_ids, decoder_generate_ids[0])
            else:
                return (input_ids, decoder_generate_ids)
    
    def compute_contrastive_loss(self, encoder_hidden_states, decoder_hidden_states):
        # encoder_hidden_states: (batch_size, seq_len, hidden_size)
        # decoder_hidden_states: (batch_size, seq_len, hidden_size)
        # attention_mask: (batch_size, seq_len)

        anchor_features = encoder_hidden_states.sum(dim=1) / encoder_hidden_states.size(1)  # (batch_size, hidden_size)
        positive_features = decoder_hidden_states.sum(dim=1) / decoder_hidden_states.size(1)  # (batch_size, hidden_size)

        npairs, hidden_size = anchor_features.shape

        similarity_function = nn.CosineSimilarity(dim=-1)

        anchor_dot_contrast = similarity_function(anchor_features.expand((npairs, npairs, hidden_size)),
                                                   positive_features.expand((npairs, npairs, hidden_size)).transpose(0, 1))  # (batch_size, batch_size)
        
        contrastive_loss_i = -nn.LogSoftmax(dim=0)(torch.div(anchor_dot_contrast, self.contrastive_temperature)).diag().sum() # (batch_size,)

        contrastive_loss_j = -nn.LogSoftmax(dim=1)(torch.div(anchor_dot_contrast, self.contrastive_temperature)).diag().sum() # (batch_size,)
        return (contrastive_loss_i + contrastive_loss_j) / 2.0 / npairs


    def compute_ot_loss_cosine(
        hidden_states_a: torch.Tensor,
        mask_a: torch.Tensor,
        hidden_states_b: torch.Tensor,
        mask_b: torch.Tensor,
        reg: float = 0.1,
        num_iters: int = 20,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Calculates the Optimal Transport (Sinkhorn) Loss between two batches of sentences 
        based on Cosine Distance. The function is highly optimized for GPU via 
        Vectorization and is absolutely safe from numerical underflow.

        Parameters:
        -----------
        hidden_states_a : torch.Tensor
            Output from the Encoder containing the semantic representations of batch A. 
            Shape: [B, T1, D] (B: Batch size, T1: Max sequence length A, D: Hidden dimension).
            Note: You do not need to normalize this tensor before passing it to the function.
            
        mask_a : torch.Tensor
            Attention mask matrix for batch A from the Tokenizer. 
            Shape: [B, T1]. 
            Values: 1 (valid token, cost will be computed), 0 (padding, will be ignored).
            
        hidden_states_b : torch.Tensor
            Output from the Encoder containing the semantic representations of batch B. 
            Shape: [B, T2, D] (T2: Max sequence length B).
            
        mask_b : torch.Tensor
            Attention mask matrix for batch B. 
            Shape: [B, T2].
            
        reg : float, default = 0.1
            Entropy regularization coefficient. Controls the "blurriness" of the Sinkhorn algorithm.
            - Too small (< 0.05): Prone to unstable/jagged gradients.
            - Too large (> 0.5): Blurs the transport matrix too much, skewing semantic alignment.
            - Recommended: 0.05 to 0.1 when using Cosine Distance.
            
        num_iters : int, default = 20
            Fixed number of iterations for the Sinkhorn algorithm to converge. 
            20 iterations is an optimal balance between computation speed and accuracy.
            
        eps : float, default = 1e-8
            A tiny value (Epsilon) added to denominators to prevent Division by Zero errors, 
            keeping the computation graph stable.

        Returns:
        --------
        torch.Tensor
            A scalar tensor representing the average Optimal Transport cost of the 
            entire Batch. You can directly call `.backward()` on it.
        """
        
        B, T1, D = hidden_states_a.shape
        T2 = hidden_states_b.shape[1]

        # 1. COMPUTE COSINE DISTANCE (COST MATRIX)
        # Normalize the vectors to length 1 (L2 Normalize)
        norm_a = F.normalize(hidden_states_a, p=2, dim=-1)
        norm_b = F.normalize(hidden_states_b, p=2, dim=-1)
        
        # Compute Cosine Similarity -> Cosine Distance. Values are bounded in [0, 2]
        cos_sim = torch.bmm(norm_a, norm_b.transpose(1, 2))
        cost = 1.0 - cos_sim

        # 2. HANDLE PADDING
        # Create a cross-attention mask between the two sentences
        valid = mask_a.unsqueeze(2) * mask_b.unsqueeze(1)
        
        # Assign a cost of 10.0 to padding positions (since the actual max cost is only 2.0)
        cost_masked = cost.masked_fill(valid == 0, 10.0)

        # 3. SINKHORN ALGORITHM
        # Compute marginal distributions (distribute mass uniformly across valid tokens)
        a = mask_a.float()
        a = a / (a.sum(dim=1, keepdim=True) + eps)

        b = mask_b.float()
        b = b / (b.sum(dim=1, keepdim=True) + eps)

        # Kernel matrix
        K = torch.exp(-cost_masked / reg)

        u = torch.ones_like(a)

        # Sinkhorn iteration loop
        for _ in range(num_iters):
            v = b / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + eps)
            u = a / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + eps)

        # Transport Plan: [B, T1, T2]
        P = u.unsqueeze(2) * K * v.unsqueeze(1)

        # 4. COMPUTE TOTAL LOSS
        # Multiply the transport plan by the original cost and only sum over valid tokens
        ot_loss = (P * cost * valid).sum(dim=(1, 2)).mean()

        return ot_loss
    
    def freeze_model(self, freeze_llm=True, freeze_decoder=True, freeze_decoder_cross_attn=True, freeze_mt_lm_head=True):
        # freeze mt encoder
        for name, param in self.mt_model.get_encoder().named_parameters():
            param.requires_grad = False
        
        # freeze mt decoder
        if freeze_decoder:
            for name, param in self.mt_model.get_decoder().named_parameters():
                param.requires_grad = False
                if 'encoder_attn' in name and not freeze_decoder_cross_attn:      # train decoder cross-attention
                    param.requires_grad = True
                    print(f"Unfroze: {name}")

        # freeze mt lm head
        if freeze_mt_lm_head:
            for name, param in self.mt_model.lm_head.named_parameters():
                param.requires_grad = False

        # freeze llm
        if freeze_llm:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False

