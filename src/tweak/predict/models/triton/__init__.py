from tweak.predict.models import ModelOutput, PredictableModel


class TritonClientModel(PredictableModel):

    def infer(self) -> ModelOutput:
        pass


class TritonClientModelForTokenClassification(TritonClientModel):

    def infer(self) -> ModelOutput:
        pass
