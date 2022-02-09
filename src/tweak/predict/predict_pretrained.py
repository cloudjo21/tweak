from tweak.predict.predictor import Predictor, PredictorConfig
from tweak.predict.toolbox import PredictionToolboxPackerForPreTrainedModel


class PreTrainedModelPredictor(Predictor):
    def __init__(self, predictor_config: PredictorConfig):
        super(PreTrainedModelPredictor, self).__init__(predictor_config)

        pred_toolbox = PredictionToolboxPackerForPreTrainedModel.pack(
            predictor_config
        )
        self.tokenizer = pred_toolbox.tokenizer
        self.model = pred_toolbox.model
        self.pred_build_klass = pred_toolbox.prediction_build_cls

    def predict(self, texts: list):
        encoded = self.tokenizer.tokenize(texts)
        output = self.model.infer(encoded)

        return self.pred_build_klass()(encoded, output)
        # return output.last_hidden_state
