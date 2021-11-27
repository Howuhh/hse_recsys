import numpy as np

NAN = "NAN"

def count_sep(string):
    seps = ("|", "&", "and", "feat", "\\", "/", ";")
    return 1 + sum(map(string.count, seps))

def bd(df):
    def map_to_cat(age):  
        # частое деление т.к. в музыке оч быстро меняется мейнстрим (такой вот inductive bias)
        if age < 5 and age > 90:
            return "outlier"
        elif 5 <= age < 10:
            return "5-10"
        elif 10 <= age < 15:
            return "10-15"
        elif 15 <= age < 20:
            return "15-20"
        elif 20 <= age < 25:
            return "20-25"
        elif 25 <= age < 30:
            return "25-30"
        elif 30 <= age < 40:
            return "30-40"
        elif 40 <= age < 50:
            return "40-50"
        elif 50 < age < 60:
            return "50-60"
        else:
            return ">60"  # старики после 60 лет слушания музыки очевидно уже оглохли

    df["bd"] = df["bd"].apply(map_to_cat).astype("category")

    return df

def gender(df):
    df["gender"] = df["gender"].cat.add_categories(NAN).fillna(NAN)
    return df

def time(df):
    df["registration_init_year"] = df["registration_init_time"].apply(lambda x: x.year).astype("category")
    df["registration_init_month"] = df["registration_init_time"].apply(lambda x: x.month).astype("category")
    df["registration_init_day"] = df["registration_init_time"].apply(lambda x: x.day).astype("category")

    df["expiration_year"] = df["expiration_date"].apply(lambda x: x.year).astype("category")
    df["expiration_month"] = df["expiration_date"].apply(lambda x: x.month).astype("category")
    df["expiration_day"] = df["expiration_date"].apply(lambda x: x.day).astype("category")

    df.drop(columns=["registration_init_time", "expiration_date"], inplace=True)

    return df

def artist(df):
    df["artist_name"] = df["artist_name"].cat.add_categories(NAN).fillna(NAN)
    df["artist_num"] = df["artist_name"].apply(count_sep).astype(np.uint8)
    return df

def composer(df):
    df["composer"] = df["composer"].cat.add_categories(NAN).fillna(NAN)
    df["composer_num"] = df["composer"].apply(count_sep).astype(np.uint8)
    return df

def lyricist(df):
    df["lyricist"] = df["lyricist"].cat.add_categories(NAN).fillna(NAN)
    df["lyricist_num"] = df["lyricist"].apply(count_sep).astype(np.uint8)
    return df

def genre(df):
    df["genre_ids"] = df["genre_ids"].cat.add_categories(NAN).fillna(NAN)
    df["genre_ids_num"] = df["genre_ids"].apply(count_sep).astype(np.uint8)
    return df

def screen_name(df):
    df["source_screen_name"] = df["source_screen_name"].cat.add_categories(NAN).fillna(NAN)
    return df

def system_tab(df):
    df["source_system_tab"] = df["source_system_tab"].cat.add_categories(NAN).fillna(NAN)
    return df

def type(df):
    df["source_type"] = df["source_type"].cat.add_categories(NAN).fillna(NAN)
    return df