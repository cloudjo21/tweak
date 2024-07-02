import torch

from tweak.predict.predictor import Predictor, PredictorConfig
from tweak.predict.toolbox import PredictionToolboxPackerForTokenClassification


class TokenClassificationPredictor(Predictor):
    def __init__(self, predictor_config: PredictorConfig):
        super(TokenClassificationPredictor, self).__init__(predictor_config)

        pred_toolbox = PredictionToolboxPackerForTokenClassification.pack(predictor_config)
        self.device = predictor_config.device
        self.tokenizer = pred_toolbox.tokenizer
        self.model = pred_toolbox.model_on_device
        self.label_list = pred_toolbox.label_list
        self.pred_build_klass = pred_toolbox.prediction_build_cls

        self.id2token = dict(map(reversed, pred_toolbox.tokenizer.tokenizer.vocab.items()))

    def predict(self, texts: list):
        """
        :param: texts        list of list => list of tokens, list => list of sentence
        """
        encoded = self.tokenizer.tokenize(texts).to(self.device)
        output = self.model.infer(encoded)

        return self.pred_build_klass()(encoded,
                                       output, 
                                       self.label_list, 
                                       encoded.get("nugget_tokens") or None, 
                                       None if encoded.get("nugget_tokens") else self.id2token)
        