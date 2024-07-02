from tweak.predict.builds import PredictionBuild
from tweak.predict.predictor import Predictor, PredictorConfig
from tweak.predict.toolbox import PredictionToolboxPackerForSeq2SeqLM


class Seq2SeqLMPredictor(Predictor):
    def __init__(self, predictor_config: PredictorConfig):
        super(Seq2SeqLMPredictor, self).__init__(predictor_config)

        pred_toolbox = PredictionToolboxPackerForSeq2SeqLM.pack(predictor_config)

        self.device = predictor_config.device
        self.tokenizer = pred_toolbox.tokenizer
        self.model = pred_toolbox.model_on_device
        self.label_list = pred_toolbox.label_list
        self.pred_build_klass = pred_toolbox.prediction_build_cls

        self.id2token = dict(map(reversed, pred_toolbox.tokenizer.tokenizer.vocab.items()))

    def predict(self, texts: list) -> PredictionBuild:
        """
        :param: texts        list of list => list of tokens, list => list of sentence
        """
        encoded = self.tokenizer.tokenizer.batch_encode_plus(texts, return_tensors="pt").to(self.device)
        # output = self.model.infer(encoded)

        if 'Token' in self.pred_build_klass.__name__:
            return self.pred_build_klass()(self.model, self.tokenizer, encoded)
        else:
            output = self.model.infer(encoded)
            return self.pred_build_klass()(encoded, output)
        