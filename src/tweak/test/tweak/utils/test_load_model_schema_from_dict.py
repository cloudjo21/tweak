import numpy as np
import orjson
import os
import pandas as pd
import pytest

from pydantic import create_model


def numpy_dtype_to_python(dtype):
    mapping = {
        np.dtype('float64'): float,
        np.dtype('float32'): float,
        np.dtype('int64'): int,
        np.dtype('int32'): int,
        np.dtype('bool'): bool,
        # Add other dtypes as needed
    }
    return mapping.get(dtype, None)


@pytest.fixture(scope="module")
def model_input_feature_dataframe():
    return pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.1, 2.2, 3.3]
        }
    )


@pytest.fixture(scope="module")
def model_input_feature_schema():
    feature_schema = None
    with open("model_input_feature_schema.json", 'r') as f:
        feature_schema = orjson.loads(f.read())
    yield feature_schema
    os.remove("model_input_feature_schema.json")


def test_it_recognizes_json_formatted_model_schema(model_input_feature_dataframe):
    feature_dtypes = model_input_feature_dataframe.loc[:, :].dtypes.to_dict()
    feature_types = dict([(k, v.name) for k, v in feature_dtypes.items()][:2])
    with open("model_input_feature_schema.json", 'w+') as f:
        f.write(orjson.dumps(feature_types).decode())

    feature_schema = None
    with open("model_input_feature_schema.json", 'r') as f:
        feature_schema = orjson.loads(f.read())
    assert "A" in feature_schema


def test_it_builds_feature_schema(model_input_feature_schema: dict):
    feature_types = model_input_feature_schema
    ModelInputFeatureSchema = create_model("inkling.ModelInputFeatureSchema", **dict([(k, (numpy_dtype_to_python(np.dtype(v)), '')) for k, v in feature_types.items()]))
    ms0 = ModelInputFeatureSchema(A=7, B=0.0)

    assert ms0.A== 7
    assert type(ms0.A) == int 
    assert ms0.B== 0.0
    assert type(ms0.B) == float


def test_it_dumps_json_schema(model_input_feature_schema: dict):
    feature_types = model_input_feature_schema
    ModelInputFeatureSchema = create_model("inkling.ModelInputFeatureSchema", **dict([(k, (numpy_dtype_to_python(np.dtype(v)), '')) for k, v in feature_types.items()]))
    ms0 = ModelInputFeatureSchema(A=7, B=0.0)

    ms0_model_dump = ms0.model_dump()
    ms0_model_dump_json = ms0.model_dump_json()

    assert type(ModelInputFeatureSchema.model_json_schema()) == dict
    assert type(ms0_model_dump) == dict
    assert type(ms0_model_dump_json) == str

    with pytest.raises(Exception):
        assert type(ms0_model_dump_json) == dict


def test_it_should_be_met_that_dictionary_feature_schema_contains_feature_schema(model_input_feature_schema: dict):

    feature_types = model_input_feature_schema
    ModelInputFeatureSchema = create_model("inkling.ModelInputFeatureSchema", **dict([(k, (numpy_dtype_to_python(np.dtype(v)), '')) for k, v in feature_types.items()]))
    ms1 = ModelInputFeatureSchema(A=7, B=0.0)

    model_input_candidate_feature_df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.1, 2.2, 3.3],
            "C": [7.7, 8.8, 9.9]
        }
    )
    feature_dtypes = model_input_candidate_feature_df.loc[:, :].dtypes.to_dict()
    feature_types = dict([(k, v.name) for k, v in feature_dtypes.items()][:2])
    ModelInputFeatureSchema2 = create_model("inkling.ModelInputFeatureSchema2", **dict([(k, (numpy_dtype_to_python(np.dtype(v)), '')) for k, v in feature_types.items()]))
    ms2 = ModelInputFeatureSchema2(A=8, B=0.5, C=9.1)

    assert set(ms1.model_dump()).issubset(set(ms2.model_dump())) is True
