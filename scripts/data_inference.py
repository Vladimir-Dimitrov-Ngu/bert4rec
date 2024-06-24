import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import lightning.pytorch
import numpy as np
import omegaconf
import polars as pl
import torch
from torch import nn

from scripts.utils import load_data


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, force_last_token_mask, conf, debug_mode=False):
        super().__init__()

        self._data = data.group_by("user_id", maintain_order=True).agg(
            pl.col("item_id")
        )
        self._force_last_token_mask = force_last_token_mask
        self.conf = conf
        self.debug_mode = debug_mode

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        row = self._data.row(index)
        track_ids = row[1][::-1]
        user_id = row[0]
        n = len(track_ids)

        input_ids = torch.IntTensor(track_ids, device="cpu")
        labels = torch.LongTensor(track_ids, device="cpu")
        mask = torch.zeros(n, device="cpu").bool()

        if self._force_last_token_mask:
            mask[0] = True
        else:
            m = min(self.conf["train"]["trim_length"], n)
            mask[:m] = torch.rand(m) < self.conf["model"]["mask_prob"]
        mask *= (
            input_ids >= self.conf["tokens"]["num_special_tokens"]
        )  # don't mask special tokens
        if mask.sum() == 0:
            mask[0] = True
            mask *= (
                input_ids >= self.conf["tokens"]["num_special_tokens"]
            )  # don't mask special tokens
        mask_indices = mask.nonzero().flatten()

        input_ids[mask_indices] = self.conf["tokens"]["mask_id"]

        labels[input_ids != self.conf["tokens"]["mask_id"]] = -100

        return input_ids, labels, user_id


class MyDataModule(lightning.pytorch.LightningDataModule):
    """A DataModule standardizes the training, val, test splits, data preparation and
    transforms. The main advantage is consistent data splits, data preparation and
    transforms across models.
    """

    def __init__(
        self,
        conf: omegaconf.dictconfig.DictConfig,
        logger: logging.Logger,
        load_data: bool = False,
        start_date: None | str = None,
        end_date: None | str = None,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.conf = conf
        self.load_data = load_data
        if self.load_data:
            self.start_date = start_date
            self.end_date = end_date
        self.logger = logger
        self.seed = seed

    def all_preprocessing_data(
        self,
        train: Optional[pl.DataFrame],
    ):
        """Preprocesses the data and prepares the datasets."""
        lightning.pytorch.seed_everything(self.seed)
        start_time = time.time()
        train = train.rename({"viewer_id": "user_id", "video_id": "item_id"})
        # Select relevant columns
        columns = [
            "event_datetime",
            "user_id",
            "item_id",
            "watch_time",
            "is_autorized",
            "video_category",
            "video_duration",
            "author_id",
        ]
        train = train[columns]
        # Filter out rows with non-positive watch time
        train = train.filter(pl.col("watch_time") > 0)
        self.logger.info(
            f"Number in observation with watch time greater than 0: {len(train)}"
        )
        # Count the number of interactions per user
        user_counts = train["user_id"].value_counts()
        self.logger.info(f"Number unique users: {len(user_counts)}")
        min_user_interactions = self.conf["data"]["min_user_interactions"]
        # Filter users with a minimum number of interactions
        users = user_counts.filter((pl.col("count") >= min_user_interactions))[
            "user_id"
        ]
        self.logger.info(
            f"Users with {min_user_interactions} or more interactions {len(users)}"
        )
        train = train.filter(pl.col("user_id").is_in(users))
        del users
        self.logger.info(
            f"Number on observation in train with users did {min_user_interactions} or more interactions {train.shape[0]}"
        )
        # Create a mapping of videos to authors
        subset = train.unique(subset="item_id")
        self.video_by_author = dict(
            zip(subset["item_id"].to_list(), subset["author_id"].to_list())
        )
        del subset
        # Define the date margin for splitting training and validation sets
        margin_str = self.conf["data"]["border_date"]
        self.logger.info(f"Margin: {margin_str}")
        margin_datetime = datetime.strptime(margin_str, "%Y-%m-%d").date()
        border_dt = margin_datetime - timedelta(days=1)
        self.logger.info(f"Border datetime {border_dt}")
        # Check if the data is defined
        if train is None:
            raise ValueError(
                "Data is undefined, please define the data or use load data mode"
            )
        # Remove timezone information from event datetime
        train = train.with_columns(
            pl.col("event_datetime").apply(lambda x: x.split("+")[0], return_dtype=str)
        )
        # Cast columns to appropriate types and sort by event datetime
        train = (
            train.cast(
                {
                    "watch_time": pl.Int32,
                    "is_autorized": pl.Boolean,
                    "video_duration": pl.Int32,
                }
            )
            .with_columns(
                pl.col("event_datetime").str.to_datetime(
                    "%Y-%m-%d %H:%M:%S", strict=True
                )
            )
            .sort("event_datetime")
        )
        # Create a previous item id column and a flag for changes in item id
        train = train.with_columns(
            prev_item_id=pl.col("item_id").shift(1).over("user_id")
        )
        train = train.with_columns(
            flag=(pl.col("item_id") != pl.col("prev_item_id")).fill_null(0)
        )
        # Create a group id for each group of consecutive same item interactions
        train = train.with_columns(group_id=pl.col("flag").cumsum().over("user_id"))
        # Aggregate interactions by user and group id, then drop the group id
        train = (
            train.group_by(("user_id", "group_id"))
            .agg(
                pl.col("event_datetime").first(),
                pl.col("item_id").first(),
                pl.col("watch_time").sum(),
                pl.col("is_autorized").first(),
                pl.col("video_category").first(),
                pl.col("video_duration").first(),
                pl.col("author_id").first(),
            )
            .drop("group_id")
            .sort("event_datetime")
        )
        # Calculate reverse position of each item in the user's sequence
        train = train.with_columns(
            r_pos=pl.col("item_id").cumcount(reverse=True).over("user_id")
        )
        # Split the data into training and validation sets based on the border date (don't use in production)
        train_history = train.filter(pl.col("event_datetime") < border_dt)
        self.logger.info(f"Train len: {len(train_history)}")
        val_labels = train.filter(
            (pl.col("r_pos") == 1) & (pl.col("event_datetime") >= border_dt)
        )
        val_history = train.filter(
            (pl.col("r_pos") > 1) & pl.col("user_id").is_in(val_labels["user_id"])
        )
        self.logger.info(f"Val history: {len(val_history)}")
        val_labels = val_labels.filter(pl.col("user_id").is_in(val_history["user_id"]))
        self.logger.info(f"Val labels: {len(val_labels)}")
        # Keep only the last `n_positions` interactions per user in the training data
        train_data = train_history.group_by("user_id").tail(
            self.conf["data"]["n_positions"]
        )
        user_counts = train_data["user_id"].value_counts()
        self.logger.info(f"Number users in train {len(user_counts)}")
        users = user_counts.filter(
            (pl.col("count") >= self.conf["data"]["min_seq_len"])
        )["user_id"]
        self.logger.info(
            f"Number users in train after filter (min_sequence_len: {self.conf['data']['min_seq_len']}): {len(users)}"
        )
        train_data = train_data.filter(pl.col("user_id").is_in(users))
        # Filter items with a minimum number of interactions
        item_counts = train_data["item_id"].value_counts()
        self.logger.info(f"Number of items: {len(item_counts)}")
        items = sorted(
            item_counts.filter((pl.col("count") >= self.conf["data"]["min_item_cnt"]))[
                "item_id"
            ]
        )
        self.logger.info(
            f"Number of items after filtering (min items count: {self.conf['data']['min_item_cnt']}): {len(item_counts)}"
        )
        # Filter validation data to include only the items in the filtered list
        val_labels = val_labels.filter(pl.col("item_id").is_in(items))
        val_history = val_history.filter(pl.col("user_id").is_in(val_labels["user_id"]))
        val_data = pl.concat([val_history, val_labels], how="diagonal")
        val_data = val_data.group_by("user_id").tail(self.conf["data"]["n_positions"])

        # Create a mapping of item ids to indices
        self.item2id = {}
        for item in items:
            self.item2id[item] = (
                len(self.item2id) + self.conf["tokens"]["num_special_tokens"]
            )
        # Map item ids in training and validation data to indices
        train_data = train_data.with_columns(
            train_data["item_id"].map_dict(
                self.item2id, default=self.conf["tokens"]["unk_id"]
            )
        )
        val_data = val_data.with_columns(
            val_data["item_id"].map_dict(
                self.item2id, default=self.conf["tokens"]["unk_id"]
            )
        )
        # Filter out users with all unknown items in their interactions
        not_unk_count = (
            train_data.with_columns(pl.col("item_id") != self.conf["tokens"]["unk_id"])
            .group_by("user_id")
            .agg(pl.col("item_id").sum())
        )
        train_data = train_data.filter(
            pl.col("user_id").is_in(
                not_unk_count.filter(pl.col("item_id") > 0)["user_id"]
            )
        )
        # Create the training and validation datasets
        self.history_data = train_data
        self.train_dataset = SeqDataset(
            train_data, True, conf=self.conf, debug_mode=False
        )
        self.val_dataset = SeqDataset(val_data, True, conf=self.conf, debug_mode=False)
        # Log the total preprocessing time
        time_all = np.round(time.time() - start_time, 2)
        self.logger.info(f"All time for preprocessing: {time_all}")
        return self.train_dataset, self.val_dataset

    def setup(self, stage: Optional[str] = None):
        """Called at the beginning of fit (train + validate), validate, test, or predict.
        This is a good hook when you need to build models dynamically or adjust something
        about them. This hook is called on every process when using DDP.
        """
        self.logger.info("SETUP DATA")
        if not hasattr(self, "val_dataset"):
            self.logger.info("No have train_dataset and val_dataset, creating...")
            if self.load_data:
                if self.end_date is None or self.start_date is None:
                    raise ValueError(
                        "Please give me start date and end date or off load mode"
                    )
                df = load_data(
                    self.start_date,
                    self.end_date,
                    out_file=self.out_file,
                    logger=self.logger,
                    pandas=False,
                )
            df = pl.read_csv(self.conf["data"]["csv_path"], ignore_errors=True)
            self.train_dataset, self.val_dataset = self.all_preprocessing_data(train=df)

    def collate_fn(self, batch):
        batch_i, batch_l, user_ids = [], [], []
        for i, l, user_id in batch:
            batch_i.append(i)
            batch_l.append(l)
            user_ids.append(user_id)
        batch_i = nn.utils.rnn.pad_sequence(
            batch_i, batch_first=True, padding_value=self.conf["tokens"]["pad_id"]
        )
        return batch_i, batch_l, user_ids

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """An iterable or collection of iterables specifying training samples"""
        if hasattr(self, "train_dataloader_data"):
            return self.train_dataloader_data
        else:
            return torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                batch_size=self.conf["data"]["batch_size"],
                num_workers=self.conf["data"]["dataloader_num_wokers"],
                drop_last=False,
                shuffle=False,
                pin_memory=False,
                collate_fn=self.collate_fn,
            )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """An iterable or collection of iterables specifying validation samples."""
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.conf["data"]["batch_size"],
            num_workers=self.conf["data"]["dataloader_num_wokers"],
            drop_last=False,
            shuffle=False,
            pin_memory=False,
            collate_fn=self.collate_fn,
        )

    def get_statistics(self):
        self.logger.info("Call get statistics method")
        if self.load_data:
            self.logger.info(f"Download data from {self.start_date} to {self.end_date}")
            if self.end_date is None or self.start_date is None:
                raise ValueError("Please give me date")
            df = load_data(
                self.start_date,
                self.end_date,
                out_file=self.out_file,
                logger=self.logger,
                pandas=False,
            )
        else:
            path = self.conf["data"]["csv_path"]
            self.logger.info(f"Read data from path: {path}")
            df = pl.read_csv(path, ignore_errors=True)
        self.train_dataset, self.val_dataset = self.all_preprocessing_data(train=df)
        self.train_loader_data = self.train_dataloader()
        return len(self.item2id), len(self.train_loader_data)

    def get_history(self, user_id):
        self.logger.info("Call get history method")
        history = (
            self.history_data.filter(pl.col("user_id") == user_id)
            .groupby("user_id", maintain_order=True)
            .agg(pl.col("item_id"), pl.col("video_category"))
        )
        return history

    def get_category_video(self, video_id):
        video_category = self.history_data.filter(pl.col("item_id") == video_id)[
            "video_category"
        ][0]
        return video_category

    def teardown(self, stage: str) -> None:
        """Called at the end of fit (train + validate), validate, test, or predict.

        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        """
        pass
