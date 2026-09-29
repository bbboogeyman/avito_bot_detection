"""Построение признаков для детектора ботов по кукам."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy

# regex для User-Agent'ов скриптов и краулеров
UA_BOT = r"headless|scrapy|curl|go-http-client|python-requests|python-urllib|node-fetch|\bbot\b|crawler|spider"


def hist_entropy(s, bins, rng):
    """Энтропия гистограммы значений (NaN, если значений меньше двух)."""
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    h, _ = np.histogram(s, bins=bins, range=rng)
    return float(sp_entropy(h)) if h.sum() else np.nan


def build_features(meta_df, ev_df):
    """Расширенный набор признаков"""
    ev_df = ev_df.copy()

    # ---------- UA ----------
    uas = pd.DataFrame({"user_agent": ev_df["user_agent"].dropna().unique()})
    low = uas["user_agent"].str.lower()
    uas["is_bot_ua"] = low.str.contains(UA_BOT, regex=True)
    uas["is_app"] = low.str.startswith("avito/")
    uas["is_mobile"] = low.str.contains("android|iphone|ipad", regex=True)
    uas["chrome_v"] = uas["user_agent"].str.extract(r"Chrome/(\d+)")[0].astype(float)
    uas["android_v"] = uas["user_agent"].str.extract(r"Android (\d+)")[0].astype(float)
    uas["ios_v"] = uas["user_agent"].str.extract(r"OS (\d+)_")[0].astype(float)
    uas["ua_len"] = uas["user_agent"].str.len()
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

    # ---------- объём и ритм ----------
    f = pd.DataFrame({
        "n_events": g.size(),
        "n_event_types": g["event_name"].nunique(),
    })
    f["span_min"] = (g["event_ts"].max() - g["event_ts"].min()).dt.total_seconds() / 60
    f["gap_median"] = g["gap_s"].median()
    f["gap_mean"] = g["gap_s"].mean()
    f["gap_std"] = g["gap_s"].std()
    f["gap_min"] = g["gap_s"].min()
    f["gap_cv"] = g["gap_s"].std() / g["gap_s"].mean().replace(0, np.nan)
    f["gap_p90"] = g["gap_s"].quantile(0.9)
    f["burst_share"] = (ev_df["gap_s"] <= 1).where(ev_df["gap_s"].notna()).groupby(ev_df["cookie_id"]).mean()
    f["gap_loose_share"] = (ev_df["gap_s"] > 60).where(ev_df["gap_s"].notna()).groupby(ev_df["cookie_id"]).mean()
    f["rate_per_min"] = f["n_events"] / f["span_min"].replace(0, np.nan)
    f["peak_per_min"] = ev_df.groupby(["cookie_id", ev_df["event_ts"].dt.floor("min")]).size() \
                             .groupby("cookie_id").max()
    f["active_minutes"] = ev_df.groupby("cookie_id")["event_ts"].apply(
        lambda s: s.dt.floor("min").nunique())
    f["minutes_share"] = f["active_minutes"] / f["span_min"].clip(lower=1)
    f["night_share"] = ev_df["event_ts"].dt.hour.between(0, 5).groupby(ev_df["cookie_id"]).mean()
    f["hour_entropy"] = ev_df.groupby("cookie_id")["event_ts"].apply(
        lambda s: float(sp_entropy(np.bincount(s.dt.hour, minlength=24) + 1e-9)))

    # доли по типам событий
    f = f.join(pd.crosstab(ev_df["cookie_id"], ev_df["event_name"], normalize="index").add_prefix("share_"))
    # абсолютные счётчики по типам
    f = f.join(pd.crosstab(ev_df["cookie_id"], ev_df["event_name"]).add_prefix("cnt_"))

    rare = ["contact_chat_open", "contact_message_sent", "contact_phone_show",
            "favorite_add", "login", "captcha_shown", "seller_page_view"]
    for r in rare:
        col = f"cnt_{r}"
        if col not in f.columns:
            f[col] = 0
    f["rare_total"] = f[[f"cnt_{r}" for r in rare]].sum(axis=1)
    f["rare_share"] = f["rare_total"] / f["n_events"].clip(lower=1)
    f["has_contact"] = (f[["cnt_contact_chat_open", "cnt_contact_message_sent",
                           "cnt_contact_phone_show"]].sum(axis=1) > 0).astype(int)
    f["has_fav"] = (f["cnt_favorite_add"] > 0).astype(int)
    f["has_login"] = (f["cnt_login"] > 0).astype(int)
    f["captcha_rate"] = f.get("cnt_captcha_shown", 0) / f["n_events"].clip(lower=1)

    # ---------- разнообразие контента ----------
    def top_share(series):
        c = series.value_counts()
        return c.iloc[0] / c.sum() if len(c) else np.nan

    def uniq_ratio(series):
        c = series.value_counts()
        return c.shape[0] / c.sum() if c.sum() else np.nan

    for col, pref in [("item_id", "item"), ("item_category", "cat"),
                      ("item_location", "loc"), ("seller_type", "seller"),
                      ("search_query", "q")]:
        s = ev_df[["cookie_id", col]].dropna()
        if len(s) == 0:
            f[f"{pref}_top_share"] = np.nan
            f[f"{pref}_uniq_ratio"] = np.nan
            continue
        gg = s.groupby("cookie_id")[col]
        f[f"{pref}_top_share"] = gg.apply(top_share)
        f[f"{pref}_uniq_ratio"] = gg.apply(uniq_ratio)

    # ---------- поиск ----------
    s_ev = ev_df[ev_df.event_name == "search_results_view"]
    s = s_ev.groupby("cookie_id").agg(
        n_search=("event_ts", "size"),
        max_page=("search_page", "max"),
        mean_page=("search_page", "mean"),
        n_queries=("search_query", "nunique"),
    )
    s["queries_per_search"] = s["n_queries"] / s["n_search"].clip(lower=1)
    s["search_deep_share"] = s_ev.assign(deep=s_ev.search_page > 3).groupby("cookie_id")["deep"].mean()

    # ---------- просмотры ----------
    v_ev = ev_df[ev_df.event_name == "item_view"]
    v = v_ev.groupby("cookie_id").agg(
        n_views=("event_ts", "size"),
        n_items=("item_id", "nunique"),
        n_categories=("item_category", "nunique"),
        n_locations=("item_location", "nunique"),
    )
    v["repeat_views"] = 1 - v["n_items"] / v["n_views"].clip(lower=1)
    v["views_per_item"] = v["n_views"] / v["n_items"].clip(lower=1)
    cat = v_ev.groupby(["cookie_id", "item_category"]).size()
    v["top_cat_share"] = cat.groupby(level=0).max() / cat.groupby(level=0).sum()

    # просмотры в уникальные минуты
    if len(v_ev):
        v["view_burst"] = v_ev.groupby(["cookie_id", v_ev["event_ts"].dt.floor("min")]).size() \
                              .groupby("cookie_id").max()

    # ---------- переходы поиск → карточка ----------
    # доля item_view, случившихся в ту же минуту, что и search_results_view
    if len(s_ev) and len(v_ev):
        srch_min = s_ev.groupby("cookie_id")["event_ts"].apply(
            lambda s: set(s.dt.floor("min")))
        def follow_ratio(sub):
            mins = srch_min.get(sub.name, set())
            return sub["event_ts"].dt.floor("min").isin(mins).mean()
        v["search_to_view_same_min"] = v_ev.groupby("cookie_id").apply(follow_ratio)

    # ---------- курсор ----------
    web_n = ev_df[ev_df.platform == "web"].groupby("cookie_id").size()
    p = ev_df[ev_df.pointer_x.notna()].copy()
    if len(p):
        pg = p.groupby("cookie_id")
        p["dx"] = pg["pointer_x"].diff()
        p["dy"] = pg["pointer_y"].diff()
        p["step"] = np.hypot(p.dx, p.dy)
        p["angle"] = np.arctan2(p.dy, p.dx)
        p["still"] = (p["step"] == 0).astype(float).where(p["step"].notna())
        lo, hi = p["dx"].quantile([0.01, 0.99])
        pg = p.groupby("cookie_id")
        ptr = pd.DataFrame({
            "ptr_n": pg.size(),
            "ptr_x_std": pg["pointer_x"].std(),
            "ptr_y_std": pg["pointer_y"].std(),
            "step_mean": pg["step"].mean(),
            "step_std": pg["step"].std(),
            "step_median": pg["step"].median(),
            "still_share": pg["still"].mean(),
            "entropy_dx": pg["dx"].apply(lambda x: hist_entropy(x, 20, (lo, hi))),
            "entropy_angle": pg["angle"].apply(lambda x: hist_entropy(x, 20, (-np.pi, np.pi))),
        })
        ptr["ptr_share"] = ptr["ptr_n"] / web_n.reindex(ptr.index)
    else:
        ptr = pd.DataFrame(index=web_n.index)

    # web-платформа без курсора - сильный сигнал
    web_ids = web_n.index
    f["web_no_ptr"] = 0
    f.loc[f.index.isin(web_ids), "web_no_ptr"] = (
        ~f.index[f.index.isin(web_ids)].isin(ptr.index if len(ptr) else [])
    ).astype(int) if len(web_ids) else 0
    # проще через reindex:
    has_ptr = pd.Series(True, index=ptr.index if len(ptr) else [])
    web_flag = pd.Series(True, index=web_ids)
    f["web_no_ptr"] = (web_flag.reindex(f.index).fillna(False) &
                       ~has_ptr.reindex(f.index).fillna(False)).astype(int)

    # ---------- UA / платформа на уровне куки ----------
    cnt = ev_df.groupby(["cookie_id", "user_agent"]).size()
    pr = cnt / cnt.groupby(level=0).transform("sum")
    ua_feat = pd.DataFrame({
        "n_ua": g["user_agent"].nunique(),
        "ua_entropy": -(pr * np.log(pr)).groupby(level=0).sum(),
        "bot_ua_share": g["is_bot_ua"].mean(),
        "mobile_share": g["is_mobile"].mean(),
        "app_share": g["is_app"].mean(),
        "chrome_ver": g["chrome_v"].median(),
        "chrome_ver_min": g["chrome_v"].min(),
        "android_ver": g["android_v"].median(),
        "ios_ver": g["ios_v"].median(),
        "ua_len_mean": g["ua_len"].mean(),
    })

    mode = lambda x: x.mode().iloc[0] if x.notna().any() and len(x.mode()) else np.nan
    cats = pd.DataFrame({
        "ua_browser": g["ua_browser"].agg(mode),
        "ua_os": g["ua_os"].agg(mode),
        "platform": g["platform"].agg(mode),
    })

    out = meta_df.set_index("cookie_id").join([f, s, v, ptr, ua_feat, cats])
    out["n_events"] = out["n_events"].fillna(0)
    out["cookie_age_h"] = (out["window_start_ts"] - out["cookie_created_at"]).dt.total_seconds() / 3600
    out["cookie_age_days"] = out["cookie_age_h"] / 24
    out["start_hour"] = out["window_start_ts"].dt.hour
    out["start_dow"] = out["window_start_ts"].dt.dayofweek
    return out
