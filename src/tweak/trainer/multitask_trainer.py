import collections
import gc
import numpy as np
import os
import torch

from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

# from transformers.trainer import Trainer
from transformers import (
    __version__,
    Trainer
)
from transformers.file_utils import is_torch_tpu_available
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    PredictionOutput,
    denumpify_detensorize,
    # has_length,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_numpify,
    nested_truncate,
)
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
)

from tunip.logger import init_logging_handler_for_klass

from tweak.dataloader.multitask_dataloader import (
    MultitaskDataLoader,
    TaskDataLoader
)
from tweak.loss.loss_computer import TripletMarginLossComputer
from tweak.metrics.metric_aggregator import WeightedMetricAggregator


# trainer
class MultitaskTrainer(Trainer):
    def __init__(self, task_name_list, mtl_data_collator, resume_from_checkpoint=None, **kwargs):
        super(MultitaskTrainer, self).__init__(**kwargs)

        self.logger = init_logging_handler_for_klass(klass=self.__class__)

        # TODO refactoring
        self.loss_computer = TripletMarginLossComputer()

        self.task_name_list = task_name_list
        self.mtl_data_collator = mtl_data_collator
        self.metric_aggregator = WeightedMetricAggregator(
            # TODO assign weight for each task
            dict([(x, 1.0 / len(self.task_name_list)) for x in self.task_name_list])
        )
        self.resume_from_checkpoint = resume_from_checkpoint

    def train(self):
        super().train(self.resume_from_checkpoint)

    def _get_single_train_dataloader(self, task_name, train_dataset):
        if self.train_dataset is None:
            raise ValueError("[MultitaskTrainer]: training requires a train_dataset")
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1  # distributed training if local_rank != -1
            else DistributedSampler(train_dataset)
        )
        data_loader = TaskDataLoader(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                # collate_fn=self.data_collator.collate_batch
                collate_fn=self.mtl_data_collator[task_name],
            ),
        )
        return data_loader

    def get_train_dataloader(self):
        return MultitaskDataLoader(
            {
                task_name: self._get_single_train_dataloader(task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            }
        )

    def _get_single_eval_dataloader(self, task_name, eval_dataset):
        if self.eval_dataset is None:
            raise ValueError("[MultitaskTrainer]: training requires a eval_dataset")
        data_loader = TaskDataLoader(
            task_name=task_name,
            data_loader=DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.mtl_data_collator[task_name],
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
            ),
        )
        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        return MultitaskDataLoader(
            {
                task_name: self._get_single_eval_dataloader(task_name, task_dataset)
                for task_name, task_dataset in (
                    eval_dataset.items()
                    if eval_dataset is not None
                    else self.eval_dataset.items()
                )
            }
        )

    def _get_single_test_dataloader(self, task_name, test_dataset):
        if not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return TaskDataLoader(
            task_name=task_name,
            data_loader=DataLoader(
                test_dataset,
                sampler=test_sampler,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.mtl_data_collator[task_name],
                drop_last=self.args.dataloader_drop_last,
            ),
        )

    def _compute_single_metrics(
        self, task_name: str, metric_key_prefix: str, output: PredictionOutput  #, inputs
    ) -> PredictionOutput:

        preds, label_ids = output.predictions, output.label_ids

        if (
            self.compute_metrics is not None
            and preds is not None
            and label_ids is not None
        ):
            metrics = self.compute_metrics[task_name](
                EvalPrediction(predictions=preds, label_ids=label_ids, inputs=None), task_name
            )
        else:
            metrics = {}

        # pass eval_loss to _compute_single_metrics
        eval_loss = output.metrics[f"{metric_key_prefix}_loss"]
        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss
            # metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    # def get_test_dataloader(self, test_dataset):
    #     # TODO
    #     pass

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        override Trainer::compute_loss
        """
        outputs = model(**inputs)
        ce_loss = outputs["loss"]

        #
        loss = ce_loss

        #
        # ex_loss = self.loss_computer(embeddings=outputs.logits, labels=inputs["labels"])
        # loss = 0.5 * ce_loss + 0.5 * ex_loss

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        if eval_dataset is not None and not isinstance(
            eval_dataset, collections.abc.Sized
        ):
            raise ValueError("eval_dataset must implement __len__")

        # eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output_dict = {}

        for task_name in self.task_name_list:
            eval_dataloader = self._get_single_eval_dataloader(
                task_name, self.eval_dataset[task_name]
            )

            output = self.evaluation_loop(
            # output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            output = self._compute_single_metrics(
                task_name, metric_key_prefix, output
            )

            self.logger.info({task_name: output.metrics})

            output_dict.update({task_name: output})

            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        # aggregate metrics from the tasks and pass it to on_evaluate()
        metric2score = dict()
        metric_task2scores = dict()
        for task_, pred_ in output_dict.items():
            for met, score in pred_.metrics.items():
                if met not in metric_task2scores:
                    metric_task2scores[met] = dict()
                metric_task2scores[met][task_] = score
        for met, task2score in metric_task2scores.items():
            metric2score[met] = self.metric_aggregator.aggregate(task2score)

        self.logger.info(metric2score)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metric2score
        )
        return metric2score
        # self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        # return output.metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, PredictionOutput]:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        .. note::

            If your predictions or labels have different sequence length (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: Dict[str, `NamedTuple`]
            key is the name of task
            A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """

        prediction_res = dict()
        for task_name in self.task_name_list:
            if test_dataset is not None and not isinstance(
                test_dataset, collections.abc.Sized
            ):
                raise ValueError("test_dataset must implement __len__")

            test_dataloader = self._get_single_test_dataloader(
                task_name, test_dataset[task_name]
            )

            prediction_res[task_name] = self.evaluation_loop(
            # prediction_res[task_name] = self.prediction_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        return prediction_res

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        self.logger.info("***** Running %s *****", description)
        self.logger.info("  Num examples = %d", num_examples)
        self.logger.info("  Batch size = %d", batch_size)
        losses_host: torch.Tensor = None
        preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None

        world_size = 1
        if is_torch_tpu_available():
            world_size = xm.xrt_world_size()
        elif self.args.local_rank != -1:
            world_size = torch.distributed.get_world_size()
        world_size = max(1, world_size)

        eval_losses_gatherer = DistributedTensorGatherer(
            world_size, num_examples, make_multiple_of=batch_size
        )
        if not prediction_loss_only:
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples)

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(
                dataloader, [self.args.device]
            ).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        for step, inputs in enumerate(dataloader):
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = (
                    losses
                    if losses_host is None
                    else torch.cat((losses_host, losses), dim=0)
                )
            if logits is not None:
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )
            if labels is not None:
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )
            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                self.args.eval_accumulation_steps is not None
                and (step + 1) % self.args.eval_accumulation_steps == 0
            ):
                eval_losses_gatherer.add_arrays(
                    self._gather_and_numpify(losses_host, "eval_losses")
                )
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(
                        self._gather_and_numpify(preds_host, "eval_preds")
                    )
                    labels_gatherer.add_arrays(
                        self._gather_and_numpify(labels_host, "eval_label_ids")
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(
            self._gather_and_numpify(losses_host, "eval_losses")
        )
        if not prediction_loss_only:
            preds_gatherer.add_arrays(
                self._gather_and_numpify(preds_host, "eval_preds")
            )
            labels_gatherer.add_arrays(
                self._gather_and_numpify(labels_host, "eval_label_ids")
            )

        eval_loss = eval_losses_gatherer.finalize()

        preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=None)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
    # ) -> EvalLoopOutput:

        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # TODO support deepspeed
        # # if eval is called w/o train init deepspeed here
        # if args.deepspeed and not self.deepspeed:

        #     # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
        #     # from the checkpoint eventually
        #     deepspeed_engine, _, _ = deepspeed_init(
        #         self, num_training_steps=0, resume_from_checkpoint=None, inference=True
        #     )
        #     self.model = deepspeed_engine.module
        #     self.model_wrapped = deepspeed_engine
        #     self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        self.logger.info(f"***** Running {description} *****")
        # if has_length(dataloader):
        #     self.logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        # else:
        #     self.logger.info("  Num examples: Unknown")
        self.logger.info(f"  Batch size = {batch_size}")

        model.eval()

        # empty cache
        self.logger.info(f"Empty Cache on {description}")
        gc.collect()
        torch.cuda.empty_cache()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # if is_torch_tpu_available():
        #     dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = inputs["input_ids"] if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # # Number of samples
        # if has_length(eval_dataset):
        #     num_samples = len(eval_dataset)
        # # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # # methods. Therefore we need to make sure it also has the attribute.
        # elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
        #     num_samples = eval_dataset.num_examples
        # else:
        #     if has_length(dataloader):
        #         num_samples = self.num_examples(dataloader)
        #     else:  # both len(dataloader.dataset) and len(dataloader) fail
        #         num_samples = observed_num_examples
        num_samples = observed_num_examples
        
        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # # Metrics!
        metrics = {}
        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        #     if args.include_inputs_for_metrics:
        #         metrics = self.compute_metrics(
        #             EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
        #         )
        #     else:
        #         metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        # else:
        #     metrics = {}

        # # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        # metrics = denumpify_detensorize(metrics)

        # if all_losses is not None:
        #     metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # # Prefix all keys with metric_key_prefix + '_'
        # for key in list(metrics.keys()):
        #     if not key.startswith(f"{metric_key_prefix}_"):
        #         metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
        return PredictionOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics)


    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if model is None:
            model = self.model

        for task_name in self.task_name_list:
            resume_from_task_checkpoint = resume_from_checkpoint + "/" + task_name

            if not os.path.isfile(os.path.join(resume_from_task_checkpoint, WEIGHTS_NAME)) and not os.path.isfile(
                os.path.join(resume_from_task_checkpoint, WEIGHTS_INDEX_NAME)
            ):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_task_checkpoint}")

            self.logger.info(f"Loading model from {resume_from_task_checkpoint}.")

            if os.path.isfile(os.path.join(resume_from_task_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(resume_from_task_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    self.logger.warning(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if self.args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            elif os.path.isfile(os.path.join(resume_from_task_checkpoint, WEIGHTS_NAME)):
                # If the model is on the GPU, it still works!
                # We load the model state dict on the CPU to avoid an OOM error.
                state_dict = torch.load(os.path.join(resume_from_task_checkpoint, WEIGHTS_NAME), map_location="cpu")
                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, task_name)
                
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)
            else:
                # We load the sharded checkpoint
                load_result = load_sharded_checkpoint(model, resume_from_task_checkpoint, strict=is_sagemaker_mp_enabled())
                if not is_sagemaker_mp_enabled():
                    self._issue_warnings_after_load(load_result)
