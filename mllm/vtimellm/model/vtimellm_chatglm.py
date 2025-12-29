import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModelForCausalLM
from .chatglm import ChatGLMConfig, ChatGLMModel, ChatGLMForConditionalGeneration
from .vtimellm_arch import VTimeLLMMetaModel, VTimeLLMMetaForCausalLM

class VTimeLLMChatGLMConfig(ChatGLMConfig):
    model_type = "VTimeLLM_ChatGLM"

class VTimeLLMChatGLMModel(ChatGLMModel, VTimeLLMMetaModel):
    config_class = VTimeLLMChatGLMConfig

    def __init__(self, config, empty_init=True, device=None):
        super(VTimeLLMChatGLMModel, self).__init__(config, empty_init=empty_init, device=device)

class VTimeLLMChatGLMForCausalLM(ChatGLMForConditionalGeneration, VTimeLLMMetaForCausalLM):
    config_class = VTimeLLMChatGLMConfig

    def __init__(self, config, empty_init=True, device=None):
        super(ChatGLMForConditionalGeneration, self).__init__(config)
        self.transformer = VTimeLLMChatGLMModel(config, empty_init=empty_init, device=device)
        self.max_sequence_length = config.max_length
        self.config = config
        self.quantized = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
        images: Optional[torch.FloatTensor] = None,
    ):

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("VTimeLLM_ChatGLM", VTimeLLMChatGLMConfig)
AutoModelForCausalLM.register(VTimeLLMChatGLMConfig, VTimeLLMChatGLMForCausalLM)

class ChatGLMForSequenceClassification(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)

        self.classifier_head = nn.Linear(config.hidden_size, config.num_labels, bias=True, dtype=torch.half)
        if config.classifier_dropout is not None:
            self.dropout = nn.Dropout(config.classifier_dropout)
        else:
            self.dropout = None
        self.config = config

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            full_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        pooled_hidden_states = hidden_states[-1]
        if self.dropout is not None:
            pooled_hidden_states = self.dropout(pooled_hidden_states)
        logits = self.classifier_head(pooled_hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze().float(), labels.squeeze())
                else:
                    loss = loss_fct(logits.float(), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.float(), labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

