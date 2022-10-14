from enum import Enum

class ContextualEmbeddingType(Enum):
    none = 0
    elmo = 1
    bert = 2 # not support yet
    flair = 3 # not support yet
