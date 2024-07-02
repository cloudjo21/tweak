import re


def _sub_sharps(token):
    return re.sub("^##", "", token)


def sub_tokens_to_words(sub_tokens, special_prefix="##"):
    words = []
    tokens = []
    for token in sub_tokens:
        if not tokens or token.startswith(special_prefix):
            tokens.append(token)
        else:
            word = ''.join(list(map(_sub_sharps, iter(tokens))))
            words.append(word)
            tokens = [token]

    if tokens:
        word = ''.join(list(map(_sub_sharps, iter(tokens))))
        words.append(word)
    
    return words


def tokens2starts(tokens):
    starts = [
        x + sum([len(w) for w in tokens[:x]]) for x in range(0, len(tokens))
    ]
    return starts


def tokens2ends(tokens):
    ends = [
        x + sum([len(w) for w in tokens[:x + 1]]) for x in range(0, len(tokens))
    ]
    return ends 


# print(tokens2starts(['the', 'end', 'game']))
# print(tokens2ends(['the', 'end', 'game']))