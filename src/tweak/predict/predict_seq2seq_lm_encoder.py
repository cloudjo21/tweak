from tweak.predict.predictor import Predictor, PredictorConfig
from tweak.predict.toolbox import PredictionToolboxPackerForSeq2SeqLMEncoder


class Seq2SeqLMEncoderPredictor(Predictor):
    def __init__(self, predictor_config: PredictorConfig):
        super(Seq2SeqLMEncoderPredictor, self).__init__(predictor_config)

        pred_toolbox = PredictionToolboxPackerForSeq2SeqLMEncoder.pack(
            predictor_config
        )
        self.tokenizer = pred_toolbox.tokenizer
        self.device = predictor_config.device
        self.model = pred_toolbox.model_on_device
        self.pred_build_klass = pred_toolbox.prediction_build_cls

    def predict(self, texts: list):
        encoded = self.tokenizer.tokenize(texts).to(self.device)
        output = self.model.infer(encoded)

        return self.pred_build_klass()(encoded, output)
