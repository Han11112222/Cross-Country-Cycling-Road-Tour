# app.py — Country Cycling Route Tracker (공식 총거리 고정 + 인증센터 기반 자동 구간계산)
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from math import radians, sin, cos, asin, sqrt

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# ───────────────────────────────────────────────────────────────────────────────
# 0) 공식 총거리(국토부/자행복 기준) — 노선 요약에 항상 이 값을 우선 표기
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
    "낙동강자전거길": 389,     # 공식 총거리
    "제주환상": 234,
}

# ───────────────────────────────────────────────────────────────────────────────
# 공용 함수
# ───────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """두 점(위경도) 사이의 대권거리(km)."""
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    R = 6371.0088  # mean Earth radius in km
    φ1, λ1, φ2, λ2 = map(radians, [lat1, lon1, lat2, lon2])
    dφ, dλ = (φ2 - φ1), (λ2 - λ1)
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

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
    need = {"category", "route", "center", "address", "lat", "lng", "id", "seq"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"centers.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    for c in ["category", "route", "center", "address", "id"]:
        df[c] = df[c].astype(str).str.strip()
    for c in ["lat", "lng", "seq"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

# 인증센터 좌표로 구간(섹션) 자동 생성
def build_sections_from_centers(centers_df: pd.DataFrame, route_name: str) -> pd.DataFrame:
    sub = centers_df[(centers_df["route"] == route_name)].dropna(subset=["lat", "lng"]).copy()
    sub = sub.sort_values("seq")
    if len(sub) < 2:
        return pd.DataFrame(columns=["category","route","section","distance_km","id"])

    rows = []
    for i in range(len(sub)-1):
        a, b = sub.iloc[i], sub.iloc[i+1]
        dist = haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])
        section = f"{int(a['seq'])}) {a['center']}→{b['center']}"
        _id = f"{route_name}@{int(a['seq'])}-{int(b['seq'])}"
        rows.append({
            "category": a["category"],
            "route": route_name,
            "section": section,
            "distance_km": round(float(dist), 1) if pd.notna(dist) else np.nan,
            "id": _id,
            "start_lat": a["lat"], "start_lng": a["lng"],
            "end_lat": b["lat"], "end_lng": b["lng"],
        })
    return pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# 2) 데이터 불러오기
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
tab = st.radio("", ["🚴 구간(거리) 추적", "📍 인증센터"], horizontal=True, label_visibility="collapsed")

# ───────────────────────────────────────────────────────────────────────────────
# 4) 구간(거리) 추적
# ───────────────────────────────────────────────────────────────────────────────
if tab == "🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")

    use_centers_auto = st.sidebar.checkbox("인증센터 좌표로 구간거리 자동 계산", value=False,
                                           help="centers.csv에 lat/lng과 seq가 있어야 합니다. 단일 노선 선택 시 동작.")

    cat_list = ["전체구간"] + sorted(routes["category"].dropna().unique().tolist())
    cat = st.sidebar.selectbox("대분류", options=cat_list, index=0)

    df = routes.copy()
    if cat != "전체구간":
        df = df[df["category"] == cat]

    route_names = sorted(df["route"].dropna().unique().tolist())
    route_pick = st.sidebar.multiselect("노선(복수 선택 가능)", route_names, default=route_names)
    if not route_pick:
        st.stop()

    # 인증센터 기반 자동 계산 (단일 노선 + centers 제공 + 좌표 있음)
    if use_centers_auto and centers is not None and len(route_pick) == 1:
        auto_df = build_sections_from_centers(centers, route_pick[0])
        if not auto_df.empty:
            df = auto_df
        else:
            st.warning("인증센터 좌표(lat/lng)가 부족해 자동 계산을 할 수 없습니다. 표의 기본 routes.csv 데이터를 사용합니다.")
            df = routes[routes["route"].isin(route_pick)].copy()
    else:
        df = df[df["route"].isin(route_pick)].copy()

    # 노선 총거리 요약(공식값 우선)
    def official_total(route: str) -> float:
        if route in OFFICIAL_TOTALS:
            return float(OFFICIAL_TOTALS[route])
        return float(routes.loc[routes["route"] == route, "distance_km"].sum())

    with st.expander("선택 노선 총거리 요약", expanded=True):
        summary = pd.DataFrame({
            "route": route_pick,
            "총거리(km)": [official_total(r) for r in route_pick]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # 진행 상태 체크
    if "done_ids" not in st.session_state:
        st.session_state.done_ids = set()
    df["완료"] = df["id"].isin(st.session_state.done_ids)

    edited = st.data_editor(
        df[["category","route","section","distance_km","완료"]],
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

    # KPI
    total_km = float(df["distance_km"].sum())
    done_km  = float(df[df["id"].isin(st.session_state.done_ids)]["distance_km"].sum())
    left_km  = max(total_km - done_km, 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    if len(route_pick) == 1:
        c4.metric("공식 노선 총거리", f"{official_total(route_pick[0]):,.1f} km")
    else:
        c4.metric("공식 노선 총거리", "다중 선택")

    # 지도 (path 또는 점)
    def parse_path(s):
        try:
            val = json.loads(s)
            if isinstance(val, list):
                return val
        except Exception:
            pass
        return None

    df["__path"] = None
    if "path" in df.columns:
        df["__path"] = df["path"].dropna().map(parse_path)

    paths = df[df["__path"].notna()].copy()

    pts = []
    for _, r in df.iterrows():
        for (lng, lat, label) in [
            (r.get("start_lng"), r.get("start_lat"), "start"),
            (r.get("end_lng"), r.get("end_lat"), "end"),
        ]:
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
        paths["__color"] = paths["id"].apply(lambda x: [28,200,138] if x in st.session_state.done_ids else [230,57,70])
        layers.append(pdk.Layer("PathLayer", paths, get_path="__path", get_color="__color",
                                width_scale=3, width_min_pixels=3, pickable=True))

    if not pts_df.empty:
        pts_df["__color"] = pts_df["done"].map(lambda b: [28,200,138] if b else [230,57,70])
        layers.append(pdk.Layer("ScatterplotLayer", pts_df,
                                get_position='[lng, lat]', get_fill_color='__color',
                                get_radius=150, pickable=True))

    view = pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text":"{name}"}),
                    use_container_width=True)

    st.caption("💡 인증센터 좌표(lat/lng)를 채우고 ‘인증센터 좌표로 구간거리 자동 계산’을 켜면, "
               "센터 간 직선거리 기반으로 섹션이 자동 생성됩니다. (정교한 경로는 path 열 JSON 사용)")

# ───────────────────────────────────────────────────────────────────────────────
# 5) 인증센터
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
    dfc = dfc[dfc["route"].isin(route_pick)].copy().sort_values(["route","seq"])

    if "done_center_ids" not in st.session_state:
        st.session_state.done_center_ids = set()

    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)
    edited = st.data_editor(
        dfc[["category","route","seq","center","address","완료"]],
        use_container_width=True, hide_index=True, key="editor_centers",
    )
    id_map = dict(zip(dfc["center"], dfc["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        _id = id_map.get(str(row["center"]))
        if _id and bool(row["완료"]):
            new_done.add(_id)
    st.session_state.done_center_ids = new_done

    # KPI
    total_cnt = int(dfc.shape[0])
    done_cnt = int(dfc[dfc["id"].isin(st.session_state.done_center_ids)].shape[0])
    left_cnt = total_cnt - done_cnt
    c1, c2, c3 = st.columns(3)
    c1.metric("선택 인증센터 수", f"{total_cnt:,}")
    c2.metric("완료한 인증센터", f"{done_cnt:,}")
    c3.metric("남은 인증센터", f"{left_cnt:,}")

    # 지도: 센터 포인트 + 순서(Path) 레이어
    geo = dfc.dropna(subset=["lat","lng"]).copy().sort_values(["route","seq"])
    if not geo.empty:
        geo["__color"] = geo["id"].apply(lambda x: [28,200,138] if x in st.session_state.done_center_ids else [230,57,70])

        # Path를 위해 route별로 경로 리스트 생성
        path_rows = []
        for rname, g in geo.groupby("route"):
            g = g.sort_values("seq")
            coords = g[["lng","lat"]].astype(float).values.tolist()
            if len(coords) >= 2:
                path_rows.append({"route": rname, "__path": coords})

        path_df = pd.DataFrame(path_rows)

        view = pdk.ViewState(latitude=float(geo["lat"].mean()), longitude=float(geo["lng"].mean()), zoom=7)

        layers = [
            pdk.Layer("ScatterplotLayer",
                      geo.rename(columns={"lat":"latitude","lng":"longitude"}),
                      get_position='[longitude, latitude]',
                      get_fill_color="__color", get_radius=180, pickable=True),
        ]
        if not path_df.empty:
            layers.append(pdk.Layer("PathLayer", path_df, get_path="__path",
                                    get_color=[120,120,255], width_scale=3, width_min_pixels=3, pickable=False))

        st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view,
                                 tooltip={"text":"{route} / {center}\n{address}"}),
                        use_container_width=True)
    else:
        st.info("좌표(lat,lng)가 비어 있는 인증센터가 있습니다. 좌표를 채우면 지도와 경로가 표시됩니다.")
