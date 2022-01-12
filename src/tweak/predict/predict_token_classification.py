from tweak.predict.predictor import Predictor, PredictorConfig
from tweak.predict.toolbox import PredictionToolboxPackerForTokenClassification


class TokenClassificationPredictor(Predictor):
    def __init__(self, predictor_config: PredictorConfig):
        super(TokenClassificationPredictor, self).__init__(predictor_config)

        pred_toolbox = PredictionToolboxPackerForTokenClassification.pack(predictor_config)
        self.tokenizer = pred_toolbox.tokenizer
        self.model = pred_toolbox.model
        self.label_list = pred_toolbox.label_list
        self.pred_build_klass = pred_toolbox.prediction_build_cls


    def predict(self, texts: list):
        """
        :param: texts        list of list => list of tokens, list => list of sentence
        """
        is_split_into_words = is_split_into_words = (
            True if isinstance(texts, list) and isinstance(texts[0], list) else False
        )
        encoded = self.tokenizer.tokenize(texts)
        output = self.model.infer(encoded)

        return self.pred_build_klass()(encoded, output, self.label_list, encoded["nugget_tokens"])
        