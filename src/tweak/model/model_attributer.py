from itertools import dropwhile


class NoPretrainedModelExistException(Exception):
    pass


class BertModelAttributer:
    @classmethod
    def has(cls, model_klass_name):
        return model_klass_name.startswith('Bert')

    @classmethod
    def get(cls, model):
        return model.bert


class RobertaModelAttributer:
    @classmethod
    def has(cls, model_klass_name):
        return model_klass_name.startswith('Roberta')

    @classmethod
    def get(cls, model):
        return model.roberta


class AlbertModelAttributer:
    @classmethod
    def has(cls, model_klass_name):
        return model_klass_name.startswith('Albert')

    @classmethod
    def get(cls, model):
        return model.albert


class BartModelAttributer:
    @classmethod
    def has(cls, model_klass_name):
        return model_klass_name.startswith('Bart')

    @classmethod
    def get(cls, model):
        return model.model


class ElectraModelAttributer:
    @classmethod
    def has(cls, model_klass_name):
        return model_klass_name.startswith('Electra')

    @classmethod
    def get(cls, model):
        return model.electra


class GPT2ModelAttributer:
    @classmethod
    def has(cls, model_klass_name):
        return model_klass_name.startswith('GPT2')

    @classmethod
    def get(cls, model):
        return model.transformer


class ModelAttributer:

    MODEL_ATTR_LIST = [
        BertModelAttributer,
        RobertaModelAttributer,
        AlbertModelAttributer,
        BartModelAttributer,
        ElectraModelAttributer,
        GPT2ModelAttributer,
    ]

    @classmethod
    def get_model(cls, model):
        model_klass_name = model.__class__.__name__

        model_attr = next(
            dropwhile(
                lambda model_attr: model_attr.has(model_klass_name) is False,
                cls.MODEL_ATTR_LIST
            )
        )

        if model_attr is not None:
            return model_attr.get(model)
        raise NoPretrainedModelExistException(f"there is no pretrained model: {model_klass_name}")
