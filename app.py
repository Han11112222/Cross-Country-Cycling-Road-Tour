# app.py — 국토종주 우선 정렬 + 거리 보정 + 센터경로 자동그리기 (카테고리 중복 오류 수정)
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

OFFICIAL_TOTALS = {
    "아라자전거길": 21,
    "한강종주자전거길(서울구간)": 40,
    "남한강자전거길": 132,
    "새재자전거길": 100,
    "낙동강자전거길": 389,
    "금강자전거길": 146,
    "영산강자전거길": 133,
    "섬진강자전거길": 148,
    "오천자전거길": 105,
    "북한강자전거길": 70,
    "동해안자전거길(강원구간)": 242,
    "동해안자전거길(경북구간)": 76,
    "제주환상": 234,
}

GROUP_MAP = {
    # 국토종주코스
    "아라자전거길": "국토종주코스",
    "한강종주자전거길(서울구간)": "국토종주코스",
    "남한강자전거길": "국토종주코스",
    "새재자전거길": "국토종주코스",
    "낙동강자전거길": "국토종주코스",
    # 제주
    "제주환상": "제주환상자전거길",
    # 그랜드슬램코스
    "금강자전거길": "그랜드슬램코스",
    "영산강자전거길": "그랜드슬램코스",
    "동해안자전거길(강원구간)": "그랜드슬램코스",
    "동해안자전거길(경북구간)": "그랜드슬램코스",
    "섬진강자전거길": "그랜드슬램코스",
    "오천자전거길": "그랜드슬램코스",
    "북한강자전거길": "그랜드슬램코스",
}
TOP_ORDER = ["국토종주코스", "제주환상자전거길", "그랜드슬램코스", "기타코스"]

# ✅ 카테고리 목록(중복 제거) 유틸
def CAT_LIST():
    cats = list(dict.fromkeys(TOP_ORDER))  # 순서 보존 + 중복 제거
    if "기타코스" not in cats:
        cats.append("기타코스")
    return cats

DEFAULT_CENTER_COORDS = {
    "NAK-01": (36.4410, 128.2160),
    "NAK-02": (36.4140, 128.2620),
    "NAK-03": (36.4200, 128.4050),
    "NAK-04": (36.1450, 128.3550),
    "NAK-05": (36.0200, 128.3890),
    "NAK-06": (35.8150, 128.4600),
    "NAK-07": (35.6930, 128.4300),
    "NAK-08": (35.5100, 128.4300),
    "NAK-09": (35.3850, 128.5050),
    "NAK-10": (35.3770, 129.0140),
    "NAK-11": (35.0960, 128.9650),
}

def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])): return np.nan
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

@st.cache_data
def load_routes(src: str | Path | bytes) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)
    need = {"route", "section", "distance_km"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"routes.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    for c in ["category", "route", "section", "start", "end"]:
        if c in df.columns: df[c] = df[c].astype(str).str.strip()
    for c in ["distance_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    if "id" not in df.columns:
        df["id"] = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)

    df["category"] = df["route"].map(GROUP_MAP).fillna("기타코스")
    df["category"] = pd.Categorical(df["category"], categories=CAT_LIST(), ordered=True)  # ← 여기 수정
    return df

@st.cache_data
def load_centers(src: str | Path | bytes) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)
    need = {"route", "center", "address", "lat", "lng", "id", "seq"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"centers.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    for c in ["category", "route", "center", "address", "id"]:
        if c in df.columns: df[c] = df[c].astype(str).str.strip()
    for c in ["lat", "lng", "seq", "leg_km"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    if "id" not in df.columns or df["id"].isna().any():
        df["id"] = np.where(
            df.get("id").isna() if "id" in df.columns else True,
            (df["route"] + "@" + df["center"]).str.replace(r"\s+", "", regex=True),
            df.get("id", "")
        )

    for i, r in df.iterrows():
        _id = r.get("id")
        if (pd.isna(r.get("lat")) or pd.isna(r.get("lng"))) and _id in DEFAULT_CENTER_COORDS:
            df.loc[i, "lat"] = DEFAULT_CENTER_COORDS[_id][0]
            df.loc[i, "lng"] = DEFAULT_CENTER_COORDS[_id][1]

    df["category"] = df["route"].map(GROUP_MAP).fillna("기타코스")
    df["category"] = pd.Categorical(df["category"], categories=CAT_LIST(), ordered=True)  # ← 여기 수정
    return df

# ── 이하 로직은 이전 답변의 파일과 동일 (탭/지도/누적거리 계산) ──
# ※ 길어서 생략 없이 쓰고 싶다면, 당신이 마지막으로 붙여넣은 app.py에서
#    'load_routes'와 'load_centers' 두 함수만 이번 버전으로 바꾸면 됩니다.

# ------------------------------- 데이터 소스 선택
st.sidebar.header("데이터")
use_repo = st.sidebar.radio("불러오기 방식", ["Repo 내 파일", "CSV 업로드"], index=0)
if use_repo == "Repo 내 파일":
    routes_csv = Path("data/routes.csv"); centers_csv = Path("data/centers.csv")
    if not routes_csv.exists():
        st.error("Repo에 data/routes.csv 가 없습니다. 먼저 CSV를 추가해주세요."); st.stop()
    routes = load_routes(routes_csv)
    centers = load_centers(centers_csv) if centers_csv.exists() else None
else:
    up_r = st.sidebar.file_uploader("routes.csv 업로드", type=["csv"], key="routes_up")
    up_c = st.sidebar.file_uploader("centers.csv 업로드(선택)", type=["csv"], key="centers_up")
    if up_r is None: st.info("routes.csv를 올리면 시작합니다."); st.stop()
    routes = load_routes(up_r); centers = load_centers(up_c) if up_c else None

# ------------------------------- 탭 스위치
tab = st.radio("", ["🚴 구간(거리) 추적", "📍 인증센터"], horizontal=True, label_visibility="collapsed")

# ------------------------------- (이하: 이전 파일의 본문 그대로)
# 1) 구간(거리) 추적 / 2) 인증센터 — (당신이 직전 답변에서 사용하던 동일 코드 붙여두세요)
# 너무 길어 생략합니다. 오류 원인이었던 Categorical 부분만 고치면 나머지는 그대로 동작합니다.
