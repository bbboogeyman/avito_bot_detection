"""Общие функции загрузки данных и сборки признаков для кейса с ботами.

Используется и в EDA, и в ноутбуке с обучением модели, чтобы признаки
train/test/eda считались буквально одним и тем же кодом.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy

TS_COLUMNS = ["cookie_created_at", "window_start_ts", "window_end_ts"]
UA_BOT = r"headless|scrapy|curl|go-http-client|python-requests|python-urllib|node-fetch|\bbot\b|crawler|spider"
PLATFORM_MAP = {"web": "web", "desktop": "web", "android": "android", "ios": "ios", "iphone": "ios"}
CAT_COLUMNS = ["ua_browser", "ua_os", "platform"]


def load_raw(data_path):
    """Читает train/test/events, приводит типы и чистит платформу и дубликаты."""
    train = pd.read_csv(data_path / "train.csv", parse_dates=TS_COLUMNS)
    test = pd.read_csv(data_path / "test.csv", parse_dates=TS_COLUMNS)
    events = pd.read_csv(data_path / "events.csv", parse_dates=["event_ts"])

    events["platform"] = events["platform"].str.lower().map(PLATFORM_MAP)
    events = events.drop_duplicates().reset_index(drop=True)
    return train, test, events


def events_in_window(train, test, events):
    """Оставляет только события внутри окна наблюдения (без before/after)."""
    meta = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    ev = events.merge(
        meta[["cookie_id", "window_start_ts", "window_end_ts", "split", "target"]], on="cookie_id", how="inner"
    )
    position = np.select(
        [ev.event_ts < ev.window_start_ts, ev.event_ts >= ev.window_end_ts],
        ["before", "after"], default="in_window",
    )
    ev = ev[position == "in_window"].reset_index(drop=True)
    return meta, ev


def hist_entropy(s, bins, rng):
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    h, _ = np.histogram(s, bins=bins, range=rng)
    return float(sp_entropy(h)) if h.sum() else np.nan


def build_features(meta_df, ev_df):
    ev_df = ev_df.copy()

    uas = pd.DataFrame({"user_agent": ev_df["user_agent"].dropna().unique()})
    low = uas["user_agent"].str.lower()
    uas["is_bot_ua"] = low.str.contains(UA_BOT, regex=True)
    uas["is_app"] = low.str.startswith("avito/")
    uas["is_mobile"] = low.str.contains("android|iphone|ipad", regex=True)
    uas["chrome_v"] = uas["user_agent"].str.extract(r"Chrome/(\d+)")[0].astype(float)
    uas["android_v"] = uas["user_agent"].str.extract(r"Android (\d+)")[0].astype(float)
    uas["ios_v"] = uas["user_agent"].str.extract(r"OS (\d+)_")[0].astype(float)
    uas["ua_os"] = np.select(
        [uas.is_app, low.str.contains("android"), low.str.contains("iphone|ipad|ios", regex=True),
         low.str.contains("windows"), low.str.contains("mac os"), low.str.contains("linux")],
        ["app", "android", "ios", "windows", "macos", "linux"], default="other")
    uas["ua_browser"] = np.select(
        [uas.is_bot_ua, uas.is_app, low.str.contains("firefox"), low.str.contains("edg/"),
         low.str.contains("chrome"), low.str.contains("safari")],
        ["script", "app", "firefox", "edge", "chrome", "safari"], default="other")
    ev_df = ev_df.merge(uas, on="user_agent", how="left")

    ev_df = ev_df.sort_values(["cookie_id", "event_ts"]).reset_index(drop=True)
    ev_df["gap_s"] = ev_df.groupby("cookie_id")["event_ts"].diff().dt.total_seconds()
    g = ev_df.groupby("cookie_id")

    f = pd.DataFrame({"n_events": g.size(), "n_event_types": g["event_name"].nunique()})
    f["span_min"] = (g["event_ts"].max() - g["event_ts"].min()).dt.total_seconds() / 60
    f["gap_median"] = g["gap_s"].median()
    f["gap_cv"] = g["gap_s"].std() / g["gap_s"].mean().replace(0, np.nan)
    f["burst_share"] = (ev_df["gap_s"] <= 1).where(ev_df["gap_s"].notna()).groupby(ev_df["cookie_id"]).mean()
    f["peak_per_min"] = ev_df.groupby(["cookie_id", ev_df["event_ts"].dt.floor("min")]).size().groupby("cookie_id").max()
    f["night_share"] = ev_df["event_ts"].dt.hour.between(0, 5).groupby(ev_df["cookie_id"]).mean()
    f = f.join(pd.crosstab(ev_df["cookie_id"], ev_df["event_name"], normalize="index").add_prefix("share_"))

    s = ev_df[ev_df.event_name == "search_results_view"].groupby("cookie_id").agg(
        n_search=("event_ts", "size"), max_page=("search_page", "max"), n_queries=("search_query", "nunique"))
    s["queries_per_search"] = s["n_queries"] / s["n_search"]

    v_ev = ev_df[ev_df.event_name == "item_view"]
    v = v_ev.groupby("cookie_id").agg(
        n_views=("event_ts", "size"), n_items=("item_id", "nunique"),
        n_categories=("item_category", "nunique"), n_locations=("item_location", "nunique"))
    v["repeat_views"] = 1 - v["n_items"] / v["n_views"]
    cat = v_ev.groupby(["cookie_id", "item_category"]).size()
    v["top_cat_share"] = cat.groupby(level=0).max() / cat.groupby(level=0).sum()

    web_n = ev_df[ev_df.platform == "web"].groupby("cookie_id").size()
    p = ev_df[ev_df.pointer_x.notna()].copy()
    pg = p.groupby("cookie_id")
    p["dx"], p["dy"] = pg["pointer_x"].diff(), pg["pointer_y"].diff()
    p["step"] = np.hypot(p.dx, p.dy)
    p["angle"] = np.arctan2(p.dy, p.dx)
    p["still"] = (p["step"] == 0).astype(float).where(p["step"].notna())
    lo, hi = p["dx"].quantile([0.01, 0.99]) if len(p) else (-1, 1)
    pg = p.groupby("cookie_id")
    ptr = pd.DataFrame({
        "ptr_n": pg.size(),
        "ptr_x_std": pg["pointer_x"].std(), "ptr_y_std": pg["pointer_y"].std(),
        "step_mean": pg["step"].mean(), "step_std": pg["step"].std(),
        "still_share": pg["still"].mean(),
        "entropy_dx": pg["dx"].apply(lambda x: hist_entropy(x, 20, (lo, hi))),
        "entropy_angle": pg["angle"].apply(lambda x: hist_entropy(x, 20, (-np.pi, np.pi))),
    })
    ptr["ptr_share"] = ptr["ptr_n"] / web_n.reindex(ptr.index)

    cnt = ev_df.groupby(["cookie_id", "user_agent"]).size()
    pr = cnt / cnt.groupby(level=0).transform("sum")
    ua = pd.DataFrame({
        "n_ua": g["user_agent"].nunique(),
        "ua_entropy": -(pr * np.log(pr)).groupby(level=0).sum(),
        "bot_ua_share": g["is_bot_ua"].mean(), "mobile_share": g["is_mobile"].mean(),
        "chrome_ver": g["chrome_v"].median(), "android_ver": g["android_v"].median(), "ios_ver": g["ios_v"].median(),
    })
    mode = lambda x: x.mode().iloc[0] if x.notna().any() else np.nan
    cats = pd.DataFrame({
        "ua_browser": g["ua_browser"].agg(mode), "ua_os": g["ua_os"].agg(mode), "platform": g["platform"].agg(mode),
    })

    out = meta_df.set_index("cookie_id").join([f, s, v, ptr, ua, cats])
    out["n_events"] = out["n_events"].fillna(0)
    out["cookie_age_h"] = (out.window_start_ts - out.cookie_created_at).dt.total_seconds() / 3600
    out["start_hour"] = out.window_start_ts.dt.hour
    out["start_dow"] = out.window_start_ts.dt.dayofweek
    return out


def feature_columns(feats):
    """Числовые и категориальные колонки-признаки (без служебных и target)."""
    num = [c for c in feats.select_dtypes("number").columns if c != "target"]
    cat = [c for c in CAT_COLUMNS if c in feats.columns]
    return num, cat
