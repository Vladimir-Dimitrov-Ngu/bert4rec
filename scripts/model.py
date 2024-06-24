import logging
from typing import Any

import lightning.pytorch as pl
import omegaconf
import torch
import transformers
from torch import nn
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup


class Bert4Rec(torch.nn.Module):
    def __init__(self, conf: omegaconf.dictconfig.DictConfig, item2id: int):
        super().__init__()
        self.conf = conf
        self.embedding_size = conf["model"]["head_size"] * conf["model"]["n_head"]
        config = BertConfig(
            vocab_size=item2id + conf["tokens"]["num_special_tokens"],
            hidden_size=self.embedding_size // conf["model"]["tiny_mode"],
            num_hidden_layers=conf["model"]["n_layer"],
            num_attention_heads=conf["model"]["n_head"],
            intermediate_size=self.embedding_size * 4,
            hidden_dropout_prob=conf["model"]["dropout"],
            attention_probs_dropout_prob=conf["model"]["dropout"],
            max_position_embeddings=conf["data"]["n_positions"],
            pad_token_id=conf["tokens"]["pad_id"],
        )
        self.padding_idx = conf["tokens"]["pad_id"]
        self.bert = BertModel(config, add_pooling_layer=False)

    def forward(self, x):
        attention_mask = x != self.padding_idx
        outputs = self.bert(x, attention_mask=attention_mask, return_dict=False)
        return outputs[0]

    def get_items_embeddings(self):
        return self.bert.embeddings.word_embeddings.weight


class MyModel(pl.LightningModule):
    def __init__(
        self,
        conf: omegaconf.dictconfig.DictConfig,
        item2id: int,
        logger: logging.Logger,
        len_train_dataloader: int | None = None,
        train_mode: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.bert4rec = Bert4Rec(conf, item2id)
        self.conf = conf
        self.my_logger = logger

        if train_mode:
            self.len_train_dataloader = len_train_dataloader
        # tokens and config
        self.num_special_tokens = conf["tokens"]["num_special_tokens"]
        self.trim_length = conf["train"]["trim_length"]
        self.vocab_size = item2id + self.num_special_tokens

        # model params and layer
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))
        self.initializer_range = 1.0 / self.bert4rec.embedding_size ** 0.5

        # loss
        loss_class_weight = torch.ones(self.vocab_size)
        loss_class_weight[: self.num_special_tokens] = 0.0
        self.loss_fn_train = torch.nn.CrossEntropyLoss(
            weight=loss_class_weight,
            reduction="none",
            label_smoothing=conf["model"]["label_smoothing"],
        )

        # metrics
        if train_mode:
            self.total = 0
            self.mrr = 0
            self.num_examples = 0
            self.hitrate = {k: 0 for k in [10, 20]}

    def forward(self, input_ids):
        return self.bert4rec(input_ids)

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        """Here you compute and return the training loss and some additional metrics
        for e.g. the progress bar or logger.
        """
        input_ids, labels = batch

        sequence_output = self.forward(input_ids)

        output_embedding_matrix = self.bert4rec.get_items_embeddings()
        prediction_scores = (
            sequence_output[:, : self.trim_length, :] @ output_embedding_matrix.T
        ) + self.bias

        prediction_scores[:, :, : self.num_special_tokens] = -10000.0

        per_example_loss = self.loss_fn_train(
            prediction_scores.view(-1, self.vocab_size),
            labels[:, : self.trim_length].flatten(),
        )
        loss = per_example_loss.mean()
        self.log(
            "train_loss", loss, logger=True, on_step=True, on_epoch=True, prog_bar=True
        )
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # TO DO
        # Можно сделать визуализацию полученных рекомендаций
        input_ids, labels = batch

        sequence_output = self.forward(input_ids)

        output_embedding_matrix = self.bert4rec.get_items_embeddings()

        prediction_scores = (
            sequence_output[:, 0, :] @ output_embedding_matrix.T
        ) + self.bias

        prediction_scores[:, : self.num_special_tokens] = -10000.0

        batch_size = input_ids.shape[0]

        per_example_loss = self.loss_fn_train(prediction_scores, labels[:, 0])
        loss = per_example_loss.sum() / batch_size
        self.log("val_loss", loss, logger=True, on_epoch=True, prog_bar=True)

        self.num_examples += batch_size
        indices = (prediction_scores * -1.0).argsort(-1)

        for i in range(batch_size):
            for j in range(10):
                if indices[i, j] == labels[i, 0]:
                    self.mrr += 1.0 / (j + 1.0)
                    self.total += 1.0
                    break
            k_values = [10, 20]
            for k in k_values:
                top_k_predictions = indices[i, :k]
                if labels[i, 0] in top_k_predictions:
                    self.hitrate[k] += 1

        return loss

    def on_validation_epoch_end(self):
        print(f"{self.mrr=} {self.total=}")
        self.log("val_mrr", self.mrr / self.num_examples, logger=True, on_epoch=True)
        self.log(
            "val_total", self.total / self.num_examples, logger=True, on_epoch=True
        )
        for k in self.hitrate:
            self.log(
                f"hitrate@{k}",
                self.hitrate[k] / self.num_examples,
                logger=True,
                on_epoch=True,
            )
        self.total = 0
        self.mrr = 0
        self.num_examples = 0
        self.hitrate = {k: 0 for k in [10, 20]}

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        """Operates on a single batch of data from the test set.
        In this step you'd normally generate examples or calculate anything of interest
        such as accuracy.
        """
        pass

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0, mode: str = "cpu"
    ) -> Any:
        """Step function called during
        :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`. By default, it
        calls :meth:`~pytorch_lightning.core.module.LightningModule.forward`.
        Override to add any processing logic.
        """
        self.my_logger.info("Predict step")
        (
            input_ids,
            _,
        ) = batch
        if mode == "cpu":
            input_ids = input_ids.to("cpu")
        else:
            input_ids = input_ids.to("cuda:0")
        sequence_output = self.forward(input_ids)
        output_embedding_matrix = self.bert4rec.get_items_embeddings()
        prediction_scores = (
            sequence_output[:, 0, :] @ output_embedding_matrix.T
        ) + self.bias
        prediction_scores[:, : self.num_special_tokens] = -10000.0
        indices = (prediction_scores * -1.0).argsort(-1)
        return indices[:, :10]

    def configure_optimizers(self):
        weight_decay = self.conf["train"]["weight_decay"]
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        param_optimizer = list(self.bert4rec.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optim = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.conf["train"]["learning_rate"]
        )
        # TO DO: add self.len_train_dataloader
        if self.conf["train"]["scheduler"] == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=1
                * self.len_train_dataloader
                // self.conf["train"]["accumulation_steps"],
                num_training_steps=10
                * self.len_train_dataloader
                // self.conf["train"]["accumulation_steps"],
            )
        elif self.conf["train"]["scheduler"] == "cosine":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optim,
                num_warmup_steps=1
                * self.len_train_dataloader
                // self.conf["train"]["accumulation_steps"],
                num_training_steps=10
                * self.len_train_dataloader
                // self.conf["train"]["accumulation_steps"],
            )
        return (
            {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "strict": True,
                },
            },
        )

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)

    def init_weights(self):
        """Initialize the weights"""
        self.my_logger.info("init weights for model")
        for module in self.children():
            if isinstance(module, nn.Linear):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                # module.weight.data.normal_(mean=0.0, std=self.initializer_range)
                module.weight.data.uniform_(
                    -self.initializer_range, self.initializer_range
                )
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def load_weights(self, weights_path):
        """Load weights from a checkpoint file."""
        self.my_logger.info(f"load weights for model from {weights_path}")
        checkpoint = torch.load(weights_path)
        self.load_state_dict(checkpoint["state_dict"])
