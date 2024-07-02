import random

from copy import deepcopy
from dataclasses import dataclass
from itertools import dropwhile, repeat
from pymonad.either import Left, Right
from typing import List, Optional

from tunip.constants import CLS, SEP

from tweak.pretrain.builders import PretrainingTaskExampleBuilder
from tweak.pretrain.examples import (
    NextSentencePredictionExample,
    PretrainingTaskExampleException,
    PretrainingTaskSubsequentTokensEmptyException
)


@dataclass
class NextSentencePredictionExampleBuildRequest:
    document_id: int
    sent_index: int
    bunch_of_sents: list # List[List[str]]
    target_n_tokens: int
    max_n_tokens: int


@dataclass
class FollowingSentenceResponse:
    tokens2: list
    next_sent_index: int
    is_next_random: bool


class NextSentencePredictionExampleBuilder(PretrainingTaskExampleBuilder):

    def __init__(self, documents):
        self.rv = random.Random(42)
        self.short_seq_prob = 0.1

        self.documents = documents
        self.n_of_documents = len(documents)

        self.sent_index = 0
    

    def __call__(self, build_req: NextSentencePredictionExampleBuildRequest) -> NextSentencePredictionExample:
        tokens1, tokens2, is_next_random = self._get_following_tokens(
            build_req.document_id,
            build_req.sent_index,
            build_req.bunch_of_sents,
            build_req.max_n_tokens,
            build_req.target_n_tokens
        )
        tokens, segments = self._get_sequential_tokens_and_segments(tokens1, tokens2)
        return NextSentencePredictionExample(tokens=tokens, segments=segments, is_next_random=int(is_next_random))
    

    @property
    def current_sent_index(self):
        return self.sent_index

    
    def _get_following_tokens(self, document_id, sent_index, bunch_of_sents, max_n_tokens, target_n_tokens):
        tokens2 = None
        n_of_failover = 0
        if len(bunch_of_sents) > 1:
            split_sent_index = self.rv.randint(1, len(bunch_of_sents) - 1)
        else:
            split_sent_index = 1
        
        trial_tokens2 = 0
        while not tokens2:
            tokens1 = [t for s in bunch_of_sents[:split_sent_index] for t in s]
            if not tokens1:
                raise PretrainingTaskExampleException(f"[tokens forward] {document_id}, {tokens1}/{bunch_of_sents}, {len(bunch_of_sents)}, {split_sent_index}")

            # chaining random_sentence and next_sentence

            fs_res = self._random_sentence(
                document_id, bunch_of_sents, target_n_tokens, sent_index, tokens1, split_sent_index
            ).either(
                lambda _: self._next_sentence(bunch_of_sents, split_sent_index, sent_index), lambda x: Right(x) 
            ).either(
                lambda _: Left(None), lambda x: Right(x) 
            ).value
            
            tokens2 = fs_res.tokens2 if fs_res else []
            if tokens2 == []:
                print("empty list at tokens2 !!!")

                trial_tokens2 += 1
                if trial_tokens2 > 2:
                    raise PretrainingTaskSubsequentTokensEmptyException

        self.sent_index = fs_res.next_sent_index

        self._truncate_input_sequence(tokens1, tokens2, max_n_tokens)

        return tokens1, tokens2, fs_res.is_next_random


    def _get_sequential_tokens_and_segments(self, tokens1: List[str], tokens2: List[str]) -> List[str]:
        assert len(tokens1) > 0 and len(tokens2) > 0

        tokens = []
        segments = []

        # +2 for CLS and SEP
        segments.extend(repeat(0, len(tokens1) + 2))
        tokens.append(CLS)
        tokens.extend(tokens1)
        tokens.append(SEP)

        # +1 for SEP
        segments.extend(repeat(1, len(tokens2) + 1))
        tokens.extend(tokens2)
        tokens.append(SEP)

        return tokens, segments


    def _random_sentence(self, document_id, bunch_of_sents, target_n_tokens, sent_index, tokens1, split_sent_index) -> Optional[FollowingSentenceResponse]:
        tokens2 = []
        is_next_random = True
        if self.rv.random() < 0.5 or len(bunch_of_sents) == 1:
            target_2_length = target_n_tokens - len(tokens1)

            rand_document_id = self.rv.randint(0, self.n_of_documents - 1)
            rand_doc_id = (list(dropwhile(lambda rand_doc_id: rand_doc_id != document_id, [self.rv.randint(0, self.n_of_documents - 1) for _ in range(5)])) or [self.rv.randint(0, self.n_of_documents - 1)])[0]
            if rand_doc_id == document_id:
                is_next_random = False
            
            rand_document = self.documents[rand_doc_id]
            rand_start = self.rv.randint(max(0, len(rand_document)-2), len(rand_document) - 1)

            if rand_start < len(rand_document):
                tokens2 = [t for d in rand_document[rand_start: len(rand_document)] for t in d]
            if not tokens2:
                return Left(PretrainingTaskSubsequentTokensEmptyException())

            n_of_rest_bunch = len(bunch_of_sents) - split_sent_index

            return Right(FollowingSentenceResponse(tokens2, sent_index - n_of_rest_bunch, is_next_random))
        return Left(None)

    
    def _next_sentence(self, bunch_of_sents, split_sent_index, sent_index) -> Optional[FollowingSentenceResponse]:
        tokens2 = []
        list(map(lambda tokens: tokens2.extend(tokens), bunch_of_sents[split_sent_index:len(bunch_of_sents)]))

        if tokens2:
            return Right(FollowingSentenceResponse(tokens2, sent_index + 1, False))
        else:
            return Left(None)


    def _truncate_input_sequence(self, tokens1, tokens2, max_n_tokens):
        while True:
            total_length = len(tokens1) + len(tokens2)
            if total_length <= max_n_tokens:
                break

            trunc_tokens = tokens1 if len(tokens1) > len(tokens2) else tokens2
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()
