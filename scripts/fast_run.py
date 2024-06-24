import datetime
import os
import warnings
from datetime import timedelta

import polars as pl
import pytorch_lightning
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup

from scripts.utils import logger_init, seed_everything

warnings.filterwarnings("ignore")

logger = logger_init("bert_fast_run.log")


logger.info(f"Random seed: {123}")
pl.set_random_seed(123)
pytorch_lightning.seed_everything(123, workers=True)
torch.set_float32_matmul_precision("high")
seed_everything(123)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_ENTITY"] = "dimitrov_rabota"
os.environ["WANDB_PROJECT"] = "train_bert4rec_model"


start_date = datetime.date(2024, 4, 23) - timedelta(days=8)
end_date = datetime.date(2024, 4, 23) - timedelta(days=1)


train = pl.read_csv(
    f"/data/vdimitrov/transformer/data/history_{start_date}_{end_date}.csv",
    ignore_errors=True,
)
train = train.with_columns(pl.col("event_datetime").apply(lambda x: x.split("+")[0]))

train = (
    train.cast(
        {"watch_time": pl.Int32, "is_autorized": pl.Boolean, "video_duration": pl.Int32}
    )
    .with_columns(
        pl.col("event_datetime").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=True)
    )
    .sort("event_datetime")
)

train = train.filter(pl.col("watch_time") > 0)
logger.info(
    f"Number in observation with watch time greater than 0: {len(train)}"
)  # 74210603 -> #166626235

train = train.rename({"viewer_id": "user_id", "video_id": "item_id"})

user_counts = train["user_id"].value_counts()
logger.info(f"Number unique users: {len(user_counts)}")
users = user_counts.filter((pl.col("count") >= 2))["user_id"]
logger.info(f"Users with 2 or more interactions {len(users)}")
train = train.filter(pl.col("user_id").is_in(users))
logger.info(
    f"Number on observation in train with users did 2 or more interactions {train.shape[0]}"
)

train = train.with_columns(prev_item_id=pl.col("item_id").shift(1).over("user_id"))
train = train.with_columns(
    flag=(pl.col("item_id") != pl.col("prev_item_id")).fill_null(0)
)
train = train.with_columns(group_id=pl.col("flag").cumsum().over("user_id"))
columns = [
    "event_datetime",
    "user_id",
    "item_id",
    "watch_time",
    "is_autorized",
    "video_category",
    "video_duration",
    "flag",
    "group_id",
    "prev_item_id",
]
train = train[columns]
train = (
    train.group_by(("user_id", "group_id"))
    .agg(
        pl.col("event_datetime").first(),
        pl.col("item_id").first(),
        pl.col("watch_time").sum(),
        pl.col("is_autorized").first(),
        pl.col("video_category").first(),
        pl.col("video_duration").first(),
    )
    .drop("group_id")
    .sort("event_datetime")
)
train = train.with_columns(
    r_pos=pl.col("item_id").cumcount(reverse=True).over("user_id")
)
border_dt = datetime.date(2024, 4, 23) - timedelta(days=2)
train_history = train.filter(pl.col("event_datetime") < border_dt)
logger.info(f"Train len: {len(train_history)}")
val_labels = train.filter(
    (pl.col("r_pos") == 1) & (pl.col("event_datetime") >= border_dt)
)
val_history = train.filter(
    (pl.col("r_pos") > 1) & pl.col("user_id").is_in(val_labels["user_id"])
)
logger.info(f"Val history: {len(val_history)}")  # 12_455_978 -> 8_159_011
val_labels = val_labels.filter(pl.col("user_id").is_in(val_history["user_id"]))
logger.info(f"Val labels: {len(val_labels)}")  # 768_714 -> 623_101

pad_id = 0
unk_id = 1
mask_id = 2
num_special_tokens = 3

n_positions = 64
trim_length = 16

mask_prob = 0.5
accumulation_steps = 1
min_seq_len = 3
min_item_cnt = 3

num_workers = 10


train_data = train_history.group_by("user_id").tail(n_positions)
user_counts = train_data["user_id"].value_counts()
logger.info(f"Number users in train {len(user_counts)}")
users = user_counts.filter((pl.col("count") >= min_seq_len))["user_id"]
logger.info(
    f"Number users in train after filter (min_sequence_len: {min_seq_len}): {len(users)}"
)  # 3_294_761 -> 3_617_895
train_data = train_data.filter(pl.col("user_id").is_in(users))
item_counts = train_data["item_id"].value_counts()
logger.info(f"Number of items: {len(item_counts)}")
items = sorted(item_counts.filter((pl.col("count") >= min_item_cnt))["item_id"])
logger.info(
    f"Number of items after filtering (min items count: {min_item_cnt}): {len(item_counts)}"
)
val_labels = val_labels.filter(pl.col("item_id").is_in(items))
val_history = val_history.filter(pl.col("user_id").is_in(val_labels["user_id"]))
val_data = pl.concat([val_history, val_labels], how="diagonal")
val_data = val_data.group_by("user_id").tail(n_positions)
item2id = {}
for item in items:
    item2id[item] = len(item2id) + num_special_tokens

train_data = train_data.with_columns(
    train_data["item_id"].map_dict(item2id, default=unk_id)
)
val_data = val_data.with_columns(val_data["item_id"].map_dict(item2id, default=unk_id))

not_unk_count = (
    train_data.with_columns(pl.col("item_id") != unk_id)
    .group_by("user_id")
    .agg(pl.col("item_id").sum())
)
train_data = train_data.filter(
    pl.col("user_id").is_in(not_unk_count.filter(pl.col("item_id") > 0)["user_id"])
)


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, force_last_token_mask):
        super().__init__()

        self._data = data.group_by("user_id", maintain_order=True).agg(
            pl.col("item_id")
        )
        self._force_last_token_mask = force_last_token_mask

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        row = self._data.row(index)
        track_ids = row[1][::-1]
        n = len(track_ids)

        input_ids = torch.tensor(track_ids).int()
        labels = torch.tensor(track_ids).long()

        mask = torch.zeros(n).bool()
        if self._force_last_token_mask:
            mask[0] = True
        else:
            m = min(trim_length, n)
            mask[:m] = torch.rand(m) < mask_prob
        mask *= input_ids >= num_special_tokens  # don't mask special tokens
        if mask.sum() == 0:
            mask[0] = True
            mask *= input_ids >= num_special_tokens  # don't mask special tokens
        mask_indices = mask.nonzero().flatten()

        input_ids[mask_indices] = mask_id

        labels[input_ids != mask_id] = -100

        return input_ids, labels


train_dataset = SeqDataset(train_data, False)
val_dataset = SeqDataset(val_data, True)


def collate_fn(batch):
    batch_i, batch_l = [], []
    for i, l in batch:
        batch_i.append(i)
        batch_l.append(l)
    batch_i = nn.utils.rnn.pad_sequence(batch_i, batch_first=True, padding_value=pad_id)
    batch_l = nn.utils.rnn.pad_sequence(batch_l, batch_first=True, padding_value=-100)
    return batch_i, batch_l


train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=256,
    num_workers=num_workers,
    drop_last=True,
    shuffle=True,
    pin_memory=False,
    collate_fn=collate_fn,
)

val_dataloader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=256,
    num_workers=num_workers,
    drop_last=False,
    shuffle=False,
    pin_memory=False,
    collate_fn=collate_fn,
)


class ItemModel(pytorch_lightning.LightningModule):
    def __init__(
        self,
        train_dataloader,
        val_dataloader,
        **kwargs,
    ):
        super().__init__()

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader

        padding_idx = kwargs.pop("padding_idx", 0)
        num_special_tokens = kwargs.pop("num_special_tokens", 3)

        n_positions = kwargs.pop("n_positions", 256)
        n_layer = kwargs.pop("n_layer", 1)
        n_head = kwargs.pop("n_head", 8)
        head_size = kwargs.pop("head_size", 64)
        embedding_size = n_head * head_size
        dropout = kwargs.pop("dropout", 0.1)

        self.padding_idx = padding_idx
        self.num_special_tokens = num_special_tokens
        self.initializer_range = 1.0 / embedding_size ** 0.5

        config = BertConfig(
            vocab_size=len(item2id) + num_special_tokens,
            hidden_size=embedding_size,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            intermediate_size=embedding_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=n_positions,
            pad_token_id=padding_idx,
        )
        self.bert = BertModel(config, add_pooling_layer=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.V = config.vocab_size

        loss_class_weight = torch.ones(self.V)
        loss_class_weight[
            :num_special_tokens
        ] = 0.0  # we don't care about a special token loss
        self.loss_fn_train = torch.nn.CrossEntropyLoss(
            weight=loss_class_weight,
            reduction="none",
            label_smoothing=kwargs.pop("label_smoothing", 0.0),
        )

        self.total = 0
        self.mrr = 0
        self.num_examples = 0
        self.hitrate = {k: 0 for k in [10, 20]}

    def forward(self, x):
        attention_mask = x != self.padding_idx
        outputs = self.bert(x, attention_mask=attention_mask, return_dict=False)
        return outputs[0]

    def get_items_embeddings(self):
        return self.bert.embeddings.word_embeddings.weight

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch

        sequence_output = self.forward(input_ids)

        output_embedding_matrix = self.get_items_embeddings()
        prediction_scores = (
            sequence_output[:, :trim_length, :] @ output_embedding_matrix.T
        ) + self.bias

        prediction_scores[:, :, : self.num_special_tokens] = -10000.0

        per_example_loss = self.loss_fn_train(
            prediction_scores.view(-1, self.V), labels[:, :trim_length].flatten()
        )
        loss = per_example_loss.mean()
        self.log("train_loss", loss, logger=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch

        sequence_output = self.forward(input_ids)

        output_embedding_matrix = self.get_items_embeddings()
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

    def configure_optimizers(self):
        weight_decay = 1e-05
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        param_optimizer = list(self.named_parameters())
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
        optim = torch.optim.AdamW(optimizer_grouped_parameters, lr=2.5e-4)
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=1 * len(train_dataloader) // accumulation_steps,
            num_training_steps=10 * len(train_dataloader) // accumulation_steps,
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

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def init_weights(self):
        """Initialize the weights"""
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
        checkpoint = torch.load(weights_path)
        self.load_state_dict(checkpoint["state_dict"])


item_model = ItemModel(
    train_dataloader,
    val_dataloader,
    padding_idx=pad_id,
    num_special_tokens=num_special_tokens,
    n_positions=n_positions,
    n_layer=4,
    n_head=8,
    head_size=32,
    dropout=0.1,
    label_smoothing=0.1,
)
# item_model.load_weights(weights_path='/data/vdimitrov/transformer/mlm/epoch=02-val_total=0.5660-mlm_v1.ckpt')
item_model.init_weights()
logger_wandb = WandbLogger(
    name="mlm_v1_experiment_2",
    project="rutube_project",
    config={
        "trim_length": trim_length,
        "mask_prob": mask_prob,
        "accumulation_steps": accumulation_steps,
        "min_seq_len": min_seq_len,
        "min_item_cnt": min_item_cnt,
    },
)

lr_monitor = LearningRateMonitor(logging_interval="step")
checkpoint_cb = ModelCheckpoint(
    dirpath="./mlm/",
    filename="{epoch:02d}-{val_total:.4f}-mlm_v1",
    monitor="val_total",
    mode="max",
    save_weights_only=True,
)

trainer = pytorch_lightning.Trainer(
    accelerator="gpu",
    logger=logger_wandb,
    callbacks=[lr_monitor, checkpoint_cb],
    max_epochs=10,
    accumulate_grad_batches=accumulation_steps,
)

trainer.fit(item_model)
