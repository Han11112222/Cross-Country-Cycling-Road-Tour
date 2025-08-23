# app.py — Country Cycling Route Tracker (공식 총거리 고정 + 인증센터 경로/누적거리)
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import math

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# ───────────────────────────────────────────────────────────────────────────────
# 0) 공식 총거리(자전거행복나눔 기준) — 노선별 총거리 표시는 이 값을 우선 사용
# ───────────────────────────────────────────────────────────────────────────────
OFFICIAL_TOTALS = {
    "아라자전거길": 21,
    "한강종주자전거길(서울구간)": 40,
    "금강자전거길": 146,
    "영산강자전거길": 133,
    "섬진강자전거길": 148,
    "오천자전거길": 105,
    "새재자전거길": 100,
    "남한강자전거길": 132,
    "북한강자전거길": 70,
    "동해안자전거길(강원구간)": 242,
    "동해안자전거길(경북구간)": 76,
    "낙동강자전거길": 389,
    "제주환상": 234,
}

# ───────────────────────────────────────────────────────────────────────────────
# 1) CSV 로더
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_routes(src: str | Path | bytes) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)
    need = {"category", "route", "section", "distance_km"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"routes.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    for c in ["category", "route", "section", "start", "end"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["distance_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "id" not in df.columns:
        df["id"] = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)

    return df


@st.cache_data
def load_centers(src: str | Path | bytes) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)
    need = {"category", "route", "center", "address", "lat", "lng"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"centers.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    # 정리
    for c in ["category", "route", "center", "address", "id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["lat", "lng", "seq"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # id 없으면 route+center로 생성
    if "id" not in df.columns or df["id"].isna().any():
        df["id"] = np.where(
            df.get("id").isna() if "id" in df.columns else True,
            (df["route"] + "@" + df["center"]).str.replace(r"\s+", "", regex=True),
            df.get("id", "")
        )
    return df


# ───────────────────────────────────────────────────────────────────────────────
# 2) 데이터 불러오기(Repo/업로드)
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("데이터")
use_repo = st.sidebar.radio("불러오기 방식", ["Repo 내 파일", "CSV 업로드"], index=0)

if use_repo == "Repo 내 파일":
    routes_csv = Path("data/routes.csv")
    centers_csv = Path("data/centers.csv")
    if not routes_csv.exists():
        st.error("Repo에 data/routes.csv 가 없습니다. 먼저 CSV를 추가해주세요.")
        st.stop()
    routes = load_routes(routes_csv)
    centers = load_centers(centers_csv) if centers_csv.exists() else None
else:
    up_r = st.sidebar.file_uploader("routes.csv 업로드", type=["csv"], key="routes_up")
    up_c = st.sidebar.file_uploader("centers.csv 업로드(선택)", type=["csv"], key="centers_up")
    if up_r is None:
        st.info("routes.csv를 올리면 시작합니다.")
        st.stop()
    routes = load_routes(up_r)
    centers = load_centers(up_c) if up_c else None

# ───────────────────────────────────────────────────────────────────────────────
# 3) 탭
# ───────────────────────────────────────────────────────────────────────────────
tab = st.radio("", ["🚴 구간(거리) 추적", "📍 인증센터"],
               horizontal=True, label_visibility="collapsed")

# ───────────────────────────────────────────────────────────────────────────────
# 4) 구간(거리) 추적 — 기존 방식 유지 + 공식 총거리 표기
# ───────────────────────────────────────────────────────────────────────────────
if tab == "🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")
    cat_list = ["전체구간"] + sorted(routes["category"].dropna().unique().tolist())
    cat = st.sidebar.selectbox("대분류", options=cat_list, index=0)

    df = routes.copy()
    if cat != "전체구간":
        df = df[df["category"] == cat]

    route_names = sorted(df["route"].dropna().unique().tolist())
    route_pick = st.sidebar.multiselect("노선(복수 선택 가능)", route_names, default=route_names)
    if not route_pick:
        st.stop()
    df = df[df["route"].isin(route_pick)].copy()

    # (A) 선택 노선 총거리 요약(공식값 우선)
    def official_total(route: str) -> float:
        return float(OFFICIAL_TOTALS.get(route, float(routes.loc[routes["route"] == route, "distance_km"].sum())))

    with st.expander("선택 노선 총거리 요약", expanded=True):
        summary = pd.DataFrame({"route": route_pick, "총거리(km)": [official_total(r) for r in route_pick]})
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # (B) 완료 체크
    if "done_ids" not in st.session_state:
        st.session_state.done_ids = set()
    df["완료"] = df["id"].isin(st.session_state.done_ids)

    edited = st.data_editor(
        df[["category", "route", "section", "distance_km", "완료"]],
        use_container_width=True, hide_index=True, key="editor_routes",
    )

    # 반영
    merge_key = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)
    id_map = dict(zip(merge_key, df["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        k = (str(row["route"]) + "@" + str(row["section"])).replace(" ", "")
        _id = id_map.get(k)
        if _id and bool(row["완료"]):
            new_done.add(_id)
    st.session_state.done_ids = new_done

    # (C) KPI(표 합계 기준)
    total_km = float(df["distance_km"].sum())
    done_km  = float(df[df["id"].isin(st.session_state.done_ids)]["distance_km"].sum())
    left_km  = max(total_km - done_km, 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("공식 노선 총거리", "다중 선택" if len(route_pick) != 1 else f"{official_total(route_pick[0]):,.1f} km")

    # (D) 지도 — path 있으면 선, 없으면 시작/끝 포인트
    def parse_path(s):
        try:
            val = json.loads(s)
            if isinstance(val, list): return val
        except Exception:
            pass
        return None

    df["__path"] = None
    if "path" in df.columns:
        df["__path"] = df["path"].dropna().map(parse_path)

    paths = df[df["__path"].notna()].copy()

    pts = []
    for _, r in df.iterrows():
        for (lng, lat, label) in [(r.get("start_lng"), r.get("start_lat"), "start"),
                                 (r.get("end_lng"), r.get("end_lat"), "end")]:
            if pd.notna(lng) and pd.notna(lat):
                pts.append({"lng": float(lng), "lat": float(lat),
                            "name": f"{r['route']} / {r['section']} ({label})",
                            "done": bool(r["id"] in st.session_state.done_ids)})
    pts_df = pd.DataFrame(pts)

    if len(pts_df) > 0:
        center_lng, center_lat = float(pts_df["lng"].mean()), float(pts_df["lat"].mean())
    else:
        center_lng, center_lat = 127.5, 36.2

    layers = []
    if not paths.empty:
        paths["__color"] = paths["id"].apply(
            lambda x: [28, 200, 138] if x in st.session_state.done_ids else [230, 57, 70]
        )
        layers.append(pdk.Layer(
            "PathLayer", paths, get_path="__path", get_color="__color",
            width_scale=3, width_min_pixels=3, pickable=True
        ))

    if not pts_df.empty:
        pts_df["__color"] = pts_df["done"].map(lambda b: [28,200,138] if b else [230,57,70])
        layers.append(pdk.Layer(
            "ScatterplotLayer", pts_df, get_position='[lng, lat]',
            get_fill_color='__color', get_radius=150, pickable=True
        ))

    st.pydeck_chart(pdk.Deck(
        layers=layers, initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7),
        tooltip={"text": "{name}"}
    ), use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# 5) 인증센터 — 표에서 체크하면 누적거리 계산 + 경로/마커 표시
# ───────────────────────────────────────────────────────────────────────────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다.")
        st.stop()

    st.sidebar.header("인증센터 필터")
    cat_list = ["전체"] + sorted(centers["category"].dropna().unique().tolist())
    cat = st.sidebar.selectbox("대분류", cat_list, index=0)

    dfc = centers.copy()
    if cat != "전체":
        dfc = dfc[dfc["category"] == cat]

    route_names = sorted(dfc["route"].dropna().unique().tolist())
    route_pick = st.sidebar.multiselect("노선(복수 선택 가능)", route_names, default=route_names)
    if not route_pick:
        st.stop()
    dfc = dfc[dfc["route"].isin(route_pick)].copy()

    # 체크 상태
    if "done_center_ids" not in st.session_state:
        st.session_state.done_center_ids = set()

    dfc = dfc.sort_values(["route", "seq", "center"], na_position="last").reset_index(drop=True)
    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)

    with st.expander("인증센터 체크(간단 편집)", expanded=True):
        edited = st.data_editor(
            dfc[["category", "route", "seq", "center", "address", "완료"]],
            use_container_width=True, hide_index=True, key="editor_centers",
        )

    # 반영
    id_map = dict(zip(dfc["center"] + "|" + dfc["route"], dfc["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        key = str(row["center"]) + "|" + str(row["route"])
        _id = id_map.get(key)
        if _id and bool(row["완료"]):
            new_done.add(_id)
    st.session_state.done_center_ids = new_done
    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)

    # 거리 계산(하버사인)
    def haversine_km(lat1, lon1, lat2, lon2):
        if any(pd.isna([lat1, lon1, lat2, lon2])): return np.nan
        R = 6371.0088
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # 노선별로 이웃센터 간 세그먼트 구성
    seg_rows = []
    for route, g in dfc.dropna(subset=["lat","lng"]).groupby("route"):
        g = g.sort_values("seq")
        centers_list = g.to_dict("records")
        for i in range(len(centers_list)-1):
            a, b = centers_list[i], centers_list[i+1]
            dist = haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])
            seg_rows.append({
                "route": route,
                "start_center": a["center"], "end_center": b["center"],
                "start_lat": a["lat"], "start_lng": a["lng"],
                "end_lat": b["lat"], "end_lng": b["lng"],
                "distance_km": dist,
                "done": bool(a["완료"] and b["완료"]),
            })
    seg_df = pd.DataFrame(seg_rows)

    # KPI — 센터 경로 기준(좌표가 있는 구간만)
    if not seg_df.empty:
        total_km_centers = float(seg_df["distance_km"].sum())
        done_km_centers  = float(seg_df.loc[seg_df["done"], "distance_km"].sum())
        left_km_centers  = max(total_km_centers - done_km_centers, 0.0)
    else:
        total_km_centers = done_km_centers = 0.0
        left_km_centers  = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 인증센터 수", f"{dfc.shape[0]:,}")
    c2.metric("완료한 인증센터", f"{int(dfc[dfc['완료']].shape[0]):,}")
    c3.metric("센터 기준 누적거리", f"{done_km_centers:,.1f} km")
    c4.metric("센터 기준 남은 거리", f"{left_km_centers:,.1f} km")

    # 지도: 경로(완료=초록, 미완료=빨강) + 센터 마커
    layers = []
    if not seg_df.empty:
        # 완료/미완료 분리
        for done_flag, color in [(True, [28,200,138]), (False, [230,57,70])]:
            src = seg_df[seg_df["done"] == done_flag].copy()
            if src.empty: continue
            src["__path"] = src.apply(lambda r: [[r["start_lng"], r["start_lat"]],[r["end_lng"], r["end_lat"]]], axis=1)
            src["__color"] = [color] * len(src)
            layers.append(pdk.Layer(
                "PathLayer", src,
                get_path="__path", get_color="__color",
                width_scale=3, width_min_pixels=3, pickable=True,
            ))

    # 센터 점
    geo = dfc.dropna(subset=["lat","lng"]).copy()
    if not geo.empty:
        geo["__color"] = geo["완료"].map(lambda b: [28,200,138] if b else [230,57,70])
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            geo.rename(columns={"lat":"latitude","lng":"longitude"}),
            get_position='[longitude, latitude]',
            get_fill_color="__color",
            get_radius=180,
            pickable=True,
        ))
        vlat, vlng = float(geo["latitude"].mean()), float(geo["longitude"].mean())
    else:
        vlat, vlng = 36.2, 127.5

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=7),
        tooltip={"text": "{route}\n{start_center} → {end_center}"}
    ), use_container_width=True)

    st.info("좌표가 비어 있는 센터가 있으면 지도 경로가 일부 나타나지 않습니다. 좌표를 채우면 센터 간 누적거리도 자동 계산됩니다.")
