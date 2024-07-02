import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    PretrainedConfig
)
from transformers.generation.utils import SampleEncoderDecoderOutput

from tweak.predict.generation.config import (
    GenerationConfig
)
from tweak.predict.generation.logits_util import (
    LogitsProcessorListFactory,
    LogitsWarperListFactory,
    StoppingCriteriaListFactory,
)


class GeneratorForLM(torch.nn.Module):
    def __init__(self, model_path, generation_config: GenerationConfig, model_config: PretrainedConfig=None):
        super(GeneratorForLM, self).__init__()
 
        self.generation_config = generation_config

        self.is_encoder_decoder = self.generation_config.is_encoder_decoder

        self.logits_procs = LogitsProcessorListFactory.create(generation_config.logits_config, generation_config.stop_config.max_length, generation_config.eos_token_id)
        self.logits_warpers = LogitsWarperListFactory.create(generation_config.logits_config)
        self.stopping_criteria = StoppingCriteriaListFactory.create(generation_config.stop_config)
        if self.generation_config.num_return_sequences == 1 and self.generation_config.logits_config.min_length == 0:
            self.gen_mode = 'greedy_search' 
        elif self.generation_config.num_return_sequences > 0 and self.generation_config.logits_config.top_k > 0.0:
            self.gen_mode = 'multinomial_sample'
        else:
            self.gen_mode = None

        config = AutoConfig.from_pretrained(model_path)
        config.output_hidden_states = self.generation_config.output_hidden_states
        config.output_attentions = self.generation_config.output_attentions
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            from_tf=False,
            config=AutoConfig.from_pretrained(model_path)
        )
        self.model.eval()


    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            # assert encoded.input_ids.device.type == self.model.device.type

            model_kwargs = dict()

            if self.generation_config.output_hidden_states and self.generation_config.is_encoder_decoder:
                model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'output_hidden_states': self.generation_config.output_hidden_states}
            else:
                model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}

            bos_token_id = self.generation_config.bos_token_id
            pad_token_id = self.generation_config.pad_token_id
            eos_token_id = self.generation_config.eos_token_id

            inputs_tensor, model_input_name, model_kwargs = self.model._prepare_model_inputs(None, bos_token_id, model_kwargs)
            batch_size = inputs_tensor.shape[0]

            # TODO is_encoder_decoder=False if causalLM
            # _prepare_attention_mask_for_generation

            # TODO decoder-only models should use left-padding for generation
            # if not self.config.is_encoder_decoder:
            #     if pad_token_id is not None and torch.sum(inputs_tensor[:, -1] == pad_token_id) > 0:
            #         logger.warning(
            #             "A decoder-only architecture is being used, but right-padding was detected! For correct "
            #             "generation results, please set `padding_side='left'` when initializing the tokenizer."
            #         )

            if self.generation_config.is_encoder_decoder:
                model_kwargs = self.model._prepare_encoder_decoder_kwargs_for_generation(
                    inputs_tensor, model_kwargs, model_input_name
                )

                input_ids = self.model._prepare_decoder_input_ids_for_generation(
                    batch_size,
                    decoder_start_token_id=None,
                    bos_token_id=bos_token_id,
                    model_kwargs=model_kwargs,
                    device=inputs_tensor.device,
                )
            else:
                # if decoder-only then inputs_tensor has to be `input_ids`
                input_ids = inputs_tensor

            if self.generation_config.num_return_sequences > 1:
                input_ids, model_kwargs = self.model._expand_inputs_for_generation(
                    input_ids=input_ids,
                    expand_size=self.generation_config.num_return_sequences, # pass num_beams or num_beams * num_return_sequences
                    is_encoder_decoder=self.is_encoder_decoder,
                    **model_kwargs
                )

            output_scores = self.generation_config.output_scores
            output_attentions = self.generation_config.output_attentions
            output_hidden_states = self.generation_config.output_hidden_states
            return_dict_in_generate = self.generation_config.return_dict_in_generate

            # init attention / hidden states / scores tuples
            scores = () if (return_dict_in_generate and output_scores) else None
            decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
            cross_attentions = () if (return_dict_in_generate and output_attentions) else None
            decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

            # TODO if support extra inputs; encoder_outputs
            # update extra outputs like encoder-(attentions/hidden_states)
            # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
            if return_dict_in_generate and self.generation_config.is_encoder_decoder:
                # encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
                encoder_attentions = model_kwargs["encoder_outputs"].attentions if output_attentions else None
                encoder_hidden_states = (
                    # model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
                    model_kwargs["encoder_outputs"].hidden_states if output_hidden_states else None
                )

            # keep track of which sequences are already finished
            unfinished_sequences = input_ids.new_tensor(input_ids.shape[0]).fill_(1)

            while True:
                # # prepare inputs
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # prediction
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=self.generation_config.output_attentions,
                    output_hidden_states=self.generation_config.output_hidden_states,
                )

                # do logits procs
                next_token_logits = outputs.logits[:, -1, :]  # last token logits
                next_token_scores = self.logits_procs(input_ids, next_token_logits)
                next_token_scores = self.logits_warpers(input_ids, next_token_scores)

                # update extra outputs like scores, (decoder/cross)-attentions, decoder hidden_states
                # Store scores, attentions and hidden_states when required
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_attentions:
                        decoder_attentions += (
                            (outputs.decoder_attentions,) if self.generation_config.is_encoder_decoder else (outputs.attentions,)
                        )
                        if self.generation_config.is_encoder_decoder:
                            cross_attentions += (outputs.cross_attentions,)

                    if output_hidden_states:
                        decoder_hidden_states += (
                            (outputs.decoder_hidden_states,)
                            if self.generation_config.is_encoder_decoder
                            else (outputs.hidden_states,)
                        )

                if self.gen_mode == 'greedy_search':
                    # argmax for greedy search
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
                else:
                    # sampling
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                # finished sentences should have their next token be a padding token
                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # concat input_ids for next prediction
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

                # TODO update token_type_ids, attention_mask for caucalLM
                self.model._update_model_kwargs_for_generation(
                    outputs, model_kwargs, is_encoder_decoder=self.generation_config.is_encoder_decoder
                )

                # if eos_token was found in one sentence, set sentence to finished
                if eos_token_id is not None:
                    unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

                # stop when each sentence is finished, or if we exceed the maximum length
                if unfinished_sequences.max() == 0 or self.stopping_criteria(input_ids, scores):
                    break
                # break

            if return_dict_in_generate:
                if self.generation_config.is_encoder_decoder:
                    if self.generation_config.use_optimize is False:
                        return SampleEncoderDecoderOutput(
                            sequences=input_ids,
                            scores=scores,
                            encoder_attentions=encoder_attentions,
                            encoder_hidden_states=encoder_hidden_states,
                            decoder_attentions=decoder_attentions,
                            cross_attentions=cross_attentions,
                            decoder_hidden_states=decoder_hidden_states,
                        )
                    else:
                        return input_ids, encoder_hidden_states, decoder_hidden_states
                else:
                    # TODO branching if use_optmizer or not for causal LM
                    return input_ids
            else:
                return input_ids
