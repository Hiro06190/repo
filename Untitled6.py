import os
import math
import warnings
from datetime import datetime, timedelta

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import streamlit as st

warnings.simplefilter("ignore")

# ----------------------------
# App settings
# ----------------------------
st.set_page_config(page_title="COVID-19 Dashboard: World Data", layout="wide")

# streamlit-folium が入っていればそれを使う（なければHTML埋め込み）
try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except Exception:
    HAS_ST_FOLIUM = False
    import streamlit.components.v1 as components

DATA_URLS = [
    "https://covid.ourworldindata.org/data/owid-covid-data.csv",
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
]
LOCAL_FALLBACK = "owid-covid-data.csv"

WORLD_GEO_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"

SPECIAL_FIT_COUNTRIES = {"Latvia", "Angola", "Bangladesh", "Kenya", "Malaysia"}

OPTIONS = {
    "Cumulative Cases": "total_cases",
    "Daily Positive Tests": "new_cases",
    "Cumluative Deaths": "total_deaths",
    "Daily Deaths": "new_deaths",
    "Reproduction Rate": "reproduction_rate",
}


# ----------------------------
# Data loading
# ----------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def fetch_data():
    errors = []

    # 1) まずオンラインを順番に試す
    for url in DATA_URLS:
        try:
            df = pd.read_csv(url)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            return df, url
        except Exception as e:
            errors.append(f"{url} -> {e}")

    # 2) 全滅したらローカルCSV
    if os.path.exists(LOCAL_FALLBACK):
        df = pd.read_csv(LOCAL_FALLBACK)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df, LOCAL_FALLBACK

    raise RuntimeError("データ取得に失敗しました:\n" + "\n".join(errors))


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_world_geo():
    return gpd.read_file(WORLD_GEO_URL)


def func(x, a, b):
    # Levy-like function
    # x > 0, b > 0 前提
    return a * np.exp(-b / x) * (b + x) / (2 * math.pi * b * x)


# ----------------------------
# Main
# ----------------------------
st.title("COVID-19 Dashboard: World Data")
st.subheader("Source: https://ourworldindata.org/coronavirus")

try:
    df, data_source = fetch_data()
    st.caption(f"Data source: {data_source}")
except Exception as e:
    st.error(f"データ読込に失敗しました。\n\n{e}")
    st.stop()

region_list = sorted(df["location"].dropna().unique().tolist())
default_region = ["Afghanistan"] if "Afghanistan" in region_list else [region_list[0]]

selected_region = st.sidebar.multiselect(
    "Select a country", region_list, default=default_region
)

if not selected_region:
    st.warning("国を1つ以上選択してください。")
    st.stop()

min_date = df.index.min().date()
max_date = df.index.max().date()

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

if start_date >= end_date:
    st.sidebar.error("Error: End date must fall after start date.")
    st.stop()
else:
    st.sidebar.success(f"Start date: `{start_date}`\n\nEnd date: `{end_date}`")

charts = st.sidebar.multiselect(
    "Select individual charts to display:",
    options=list(OPTIONS.keys()),
    default=list(OPTIONS.keys())[0:1],
)

if not charts:
    st.warning("表示するチャートを1つ以上選択してください。")
    st.stop()

md = """
<p style="font-family:Courier; color:Green; font-size: 20px;">
The fitting functions explained in my attached note will be plotted, when Latvia, Angola,
Bangladesh, Kenya or Malaysia is selected in the selection box ("Select a country")
</p>
"""
st.markdown(md, unsafe_allow_html=True)

# ----------------------------
# Plot line charts
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

# fitting期間（元コード準拠）
fit_start = pd.Timestamp(datetime(2020, 4, 1))
fit_end = pd.Timestamp(datetime(2021, 1, 31))

for region in selected_region:
    df_region = df[df["location"] == region].copy()
    if df_region.empty:
        continue

    # 表示期間
    df_plot = df_region.loc[start_ts:end_ts]

    for chart in charts:
        col = OPTIONS[chart]
        if col not in df_plot.columns:
            continue

        s = df_plot[col].dropna()
        if not s.empty:
            ax.plot(s.index, s.values, label=f"{region}_{chart}")

        # 特定国かつCumulative Casesだけフィッティング描画
        if region in SPECIAL_FIT_COUNTRIES and chart == "Cumulative Cases":
            fit_series = df_region.loc[fit_start:fit_end - pd.Timedelta(days=1), "total_cases"].dropna()
            if len(fit_series) < 10:
                continue

            N = fit_series.values.astype(float)
            xx = np.arange(1, len(N) + 1, dtype=float)

            try:
                # bが0にならないよう制約
                params, _ = scipy.optimize.curve_fit(
                    func,
                    xx,
                    N,
                    p0=(max(np.nanmax(N), 1.0), 10.0),
                    bounds=([0.0, 1e-6], [np.inf, np.inf]),
                    maxfev=20000,
                )
                A, B = float(params[0]), float(params[1])
                C = float(N[0])
                y2 = C + func(xx, A, B)

                fit_dates = pd.date_range(fit_start, periods=len(y2), freq="D")
                ax.plot(fit_dates, y2, label=f"fitting_by_stochastic_model_{region}_{chart}")
            except Exception:
                # フィッティング失敗時はスキップ
                pass

ax.set_xlabel("Date")
ax.set_xlim(start_ts, end_ts)
ax.legend(loc="upper left", fontsize=8)
st.pyplot(fig)

# ----------------------------
# Reproduction rate world map
# ----------------------------
md1 = """
<p style="font-family:Courier; color:Green; font-size: 20px;">
The world map with reproduction rate (using the data of end date (of 5days ago) on side bar).
The countries with the reproduction rate ≥ 1 are colored in red, while those with the reproduction
rate < 1 are colored in blue.
</p>
"""
st.markdown(md1, unsafe_allow_html=True)

target_date = pd.Timestamp(end_date) - pd.Timedelta(days=5)

# まず target_date ぴったりを探す
df3 = df[df.index.normalize() == target_date.normalize()].copy()

# なければ target_date 以前の最新日を使う
if df3.empty:
    past = df[df.index <= target_date]
    if not past.empty:
        nearest_date = past.index.max().normalize()
        df3 = df[df.index.normalize() == nearest_date].copy()
        st.info(f"{target_date.date()} のデータがないため、直近の {nearest_date.date()} を使用しています。")
    else:
        df3 = pd.DataFrame(columns=df.columns)

if df3.empty:
    st.warning("地図表示に必要なデータが見つかりませんでした。")
    st.stop()

df3 = df3[["location", "reproduction_rate"]].dropna()
df3A = df3[df3["reproduction_rate"] >= 1]
df3B = df3[df3["reproduction_rate"] < 1]

try:
    gdf = fetch_world_geo().copy()
except Exception as e:
    st.error(f"世界地図データの取得に失敗しました: {e}")
    st.stop()

# 地図側の国名カラムを自動判定
name_col = next((c for c in ["name", "NAME", "admin", "ADMIN"] if c in gdf.columns), None)
if name_col is None:
    st.error(f"国名カラムが見つかりません。gdf columns: {list(gdf.columns)}")
    st.stop()

gdf["Group"] = "No data"
gdf.loc[gdf[name_col].isin(df3A["location"]), "Group"] = "Reproduction rate≥1"
gdf.loc[gdf[name_col].isin(df3B["location"]), "Group"] = "Reproduction rate<1"

m = folium.Map(location=[0, 0], zoom_start=2, tiles="cartodbpositron")

def add_geo_layer(map_obj, gdf_layer, fill_color, line_color):
    """空レイヤーにはGeoJsonTooltipを付けずにスキップ"""
    if gdf_layer.empty:
        return False

    tooltip_fields = [name_col, "Group"]
    tooltip = folium.features.GeoJsonTooltip(
        fields=tooltip_fields, labels=True, sticky=True
    )

    folium.GeoJson(
        gdf_layer,
        style_function=lambda x: {
            "fillColor": fill_color,
            "color": line_color,
            "weight": 1,
            "fillOpacity": 0.6,
        },
        tooltip=tooltip,
    ).add_to(map_obj)
    return True

added_red = add_geo_layer(
    m,
    gdf[gdf["Group"] == "Reproduction rate≥1"],
    fill_color="red",
    line_color="red",
)
added_blue = add_geo_layer(
    m,
    gdf[gdf["Group"] == "Reproduction rate<1"],
    fill_color="blue",
    line_color="blue",
)

# どちらも空なら、No dataだけ表示（tooltipなし）
if not added_red and not added_blue:
    st.warning("選択日の reproduction_rate データが地図にマッチしませんでした（No dataのみ表示）。")
    folium.GeoJson(
        gdf,
        style_function=lambda x: {
            "fillColor": "lightgray",
            "color": "gray",
            "weight": 0.5,
            "fillOpacity": 0.3,
        },
    ).add_to(m)

if HAS_ST_FOLIUM:
    st_folium(m, width=1100, height=550)
else:
    components.html(m._repr_html_(), height=600)

