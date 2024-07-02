import pyarrow as pa
import numpy as np

from typing import Tuple


class VectorLoader:

    def __init__(self, vectors_path, id_field_index=0, content_field_index=1):
        self.path = vectors_path
        self.id_field_index = id_field_index
        self.content_field_index = content_field_index

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        with pa.memory_map(self.path, "rb") as source:
            vectors = pa.ipc.open_file(source).read_all()
        return (vectors[self.id_field_index], vectors[self.content_field_index].to_numpy())
