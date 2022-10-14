import random

from itertools import filterfalse

from tunip.constants import CLS, MASK, SEP

from tweak.pretrain.builders import PretrainingTaskExampleBuilder
from tweak.pretrain.examples import MaskedLanguageModelExample, MaskedLanguageModelToken


class MaskedLanguageModelExampleBuilder(PretrainingTaskExampleBuilder):

    def __init__(self, max_predictions_per_sent, vocab_tokens):
        self.rv = random.Random(42)
        self.max_predictions_per_sent = max_predictions_per_sent
        self.vocab_tokens = vocab_tokens

        self.masked_lm_prob = 0.15
        self.masked_token_prob = 0.8
        self.original_token_prob = 0.5


    def __call__(self, tokens):
        token_indices = [i for i, _ in enumerate(filterfalse(lambda token: token in [CLS, SEP], tokens))]
        self.rv.shuffle(token_indices)

        num_to_predict = min(self.max_predictions_per_sent, max(1, int(round(len(tokens) * self.masked_lm_prob))))

        masked_tokens = []
        count_to_predict = 0
        for index in token_indices:
            predict_mask = count_to_predict < num_to_predict
            if predict_mask:
                masked_token_lexical = self._create_masked_token(tokens[index]) 
                count_to_predict += 1
            else:
                masked_token_lexical = tokens[index]

            masked_tokens.append(
                MaskedLanguageModelToken(index=index, input_token=tokens[index], output_label=masked_token_lexical, predict_mask=predict_mask)
            )
        example = MaskedLanguageModelExample(tokens=masked_tokens)
        example.sort()
        return example

    
    def _create_masked_token(self, token):

        # 80% masking token
        if self.rv.random() < self.masked_token_prob:
            masked_token = MASK
        # 10% original token
        elif self.rv.random() < self.original_token_prob:
            masked_token = token
        # 10% replaced with random token
        else:
            masked_token = self.vocab_tokens[self.rv.randint(0, len(self.vocab_tokens) - 1)]

        return masked_token
