

class HFAutoModel(Model):

    def __init__(self, model_config: ModelConfig):
        self.model_dir = f"{config.model_path}/{config.checkpoint}/{task_name}" if config.checkpoint else config.model_path

        self.auto_config = AutoConfig.from_pretrained(
            self.model_dir, finetuning_task=config.task_name
        )
        # self.pt_model_name = auto_config._name_or_path

        
    def infer(self) -> ModelOutput:
        pass


class HFAutoModelForTokenClassification(HFAutoModel):

    def __init__(self, model_config: ModelConfig):
        super(HFAutoModelForTokenClassification, self).__init__(self, model_config)

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_dir,
            from_tf=False,
            config=self.auto_config,
        )
        self.model.eval()
    

    def infer(self) -> ModelOutput:
        pass
