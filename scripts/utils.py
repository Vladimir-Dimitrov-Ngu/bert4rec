import logging
import os
import random
import time
from datetime import timedelta

import clickhouse_connect
import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning
import torch
from tqdm import tqdm


def logger_init(file_path: str = "log.log"):
    logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def load_data(start_date, end_date, out_file, logger, pandas=True):
    logger.info("Init data loading...")
    start = time.time()
    # If needed file alredy exists
    if os.path.isfile(out_file):
        logger.info("The user data already exists. Load it from filesystem")
        if pandas:
            loaded_dataset = pd.read_csv(out_file, dtype={"viewer_id": str})
        else:
            loaded_dataset = pl.read_csv(out_file, dtypes={"viewer_id": str})
        return loaded_dataset

    logger.info("Connect to ClickHouse...")
    client = clickhouse_connect.get_client(
        host="10.66.14.210",
        port=9090,
        username="ML_user",
        password="Ge8vMrXKiDDun8V5",
        connect_timeout=100,
        send_receive_timeout=1800,
    )

    dd = [
        str(start_date + timedelta(days=x)) for x in range((end_date - start_date).days)
    ]
    logger.info(f"Download data for {len(dd)} days...")
    for i, day in tqdm(enumerate(dd), desc="load data"):
        logger.info("Start loading new day...")
        logger.info(f"Day: {day}")

        result = client.query_df(
            f"""
            select
                    viewer_uid,
                    session_id,
                    ydevid,
                    if(viewer_uid = 0, if(match(ydevid, '[0-9]'), ydevid, session_id), toString(viewer_uid)) viewer_id,
                    rutube_video_id video_id,
                    dateAdd(hour, -3, event_timestamp) event_datetime,
                    watchtime watch_time,
                    dictGet('ruform_event_rutube_video', 'user_id', tuple(rutube_video_id)) author_id,
                    dictGet('ruform_event_rutube_video', 'duration', tuple(rutube_video_id)) video_duration,
                    dictGet('ruform_event_rutube_video_category','name', toUInt64(dictGet('ruform_event_rutube_video', 'category_id', tuple(rutube_video_id)))) video_category,
                    if(match(ydevid, '[0-9]'), 'app', 'web') platform,
                    if(viewer_uid = 0, 0, 1) is_autorized
            from ruform.events
            where 1=1
            and event_date between '{day}' and '{day}'
            and event_type = 'video/start'
            and author_id not in (23723926,23723928,23723931)
            """
        )

        logger.info(f"Num rows: {result.shape} for {day}")
        if i == 0:
            result.to_csv(out_file, mode="w", header=True, index=False)
        else:
            result.to_csv(out_file, mode="a", header=False, index=False)
    if pandas:
        loaded_dataset = pd.read_csv(out_file, dtype={"viewer_id": str})
    else:
        loaded_dataset = pl.read_csv(out_file, dtypes={"viewer_id": str})
    logger.info(f"Time to load the data: {time.time() - start}")
    return loaded_dataset


def seed_everything(seed=42):
    pl.set_random_seed(seed)
    pytorch_lightning.seed_everything(seed, workers=True)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
