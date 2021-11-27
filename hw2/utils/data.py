import os
import numpy as np
import pandas as pd
import utils.features as features

from copy import copy


MEMBERS_CONFIG = {
    "dtype": {
        "msno": "category",
        "city": "category",
        "bd": np.int32,
        "gender": "category",
        "registred_via": "category"
    },
    "parse_dates": ["registration_init_time", "expiration_date"]
}

SONGS_CONFIG = {
    "dtype": {
        "song_id": "category",
        "song_length": np.int32,
        "genre_ids": "category",
        "artist_name": "category",
        "composer": "category",
        "lyricist": "category",
        "language": "category",
    }
}

TRAIN_CONFIG = {
    "dtype": {
        "msno": "category",
        "song_id": "category",
        "source_system_tab": "category",
        "source_screen_name": "category",
        "source_type": "category",
        "target": np.int32,
    }
}

TEST_CONFIG = copy(TRAIN_CONFIG)
TEST_CONFIG["dtype"].pop("target")

EXTRA_CONFIG = {
    "usecols": ["song_id", "name"],
    "dtype": {"song_id": "category", "name": "category"}
}

def preprocess_df(df, f_map):
    for col, f in f_map:
        if col in df.columns:
            df = f(df)
    return df


# <UNK>

def load_data(
    data_path, load_test=False,
    member_config=MEMBERS_CONFIG, 
    songs_config=SONGS_CONFIG, 
    train_config=TRAIN_CONFIG, 
    extra_config=EXTRA_CONFIG
    ):
    # preprocessing functions mapping
    members_map = [
        ("bd", features.bd),
        ("gender", features.gender),
        ("registration_init_time", features.time)
    ]
    songs_map = [
        ("artist_name", features.artist),
        ("composer", features.composer),
        ("lyricist", features.lyricist),
        ("genre_ids", features.genre)
    ]
    train_map = [
        ("source_screen_name", features.screen_name),
        ("source_system_tab", features.system_tab),
        ("source_type", features.type)
    ]

    # data loading and features preprocessing
    train_df = preprocess_df(
        pd.read_csv(os.path.join(data_path, "train.csv"), **train_config),
        f_map=train_map
    )
    members_df = preprocess_df(
        pd.read_csv(os.path.join(data_path, "members.csv"), **member_config), 
        f_map=members_map
    )
    songs_df = preprocess_df(
        pd.read_csv(os.path.join(data_path, "songs.csv"), **songs_config) \
            .merge(
                pd.read_csv(os.path.join(data_path, "song_extra_info.csv"), **extra_config),
                how="left", on="song_id"
            ),
        f_map=songs_map
    )

    # merge all additional datasets to train
    # шок, пандас при мердже кастит все к float64 потому что nan это float64, какой идиот это придумал
    train = train_df \
        .merge(members_df, on="msno", how="left") \
        .merge(songs_df, on="song_id", how="left") \
        .dropna() \
        .astype({"msno": "category", "song_id": "category"})
    
    return train