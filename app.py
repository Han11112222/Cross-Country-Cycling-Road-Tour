# app.py — 공식거리 자동 보정 + 인증센터 기반 경로/거리 생성 (선택 기본값 해제 + 전체선택/해제 버튼)
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# ───────────────────────────────────────────────────────────────────────────────
# 0) 공식 총거리(자전거행복나눔 기준) — 표/누적 계산에 사용(거리 0일 때 자동 대체)
# ───────────────────────────────────────────────────────────────────────────────
OFFICIAL_TOTALS = {
    # 국토종주
    "아라자전거길": 21,
    "한강종주자전거길(서울구간)": 40,
    "남한강자전거길": 132,
    "새재자전거길": 100,
    "낙동강자전거길": 389,
    # 그랜드슬램
    "금강자전거길": 146,
    "영산강자전거길": 133,
    "북한강자전거길": 70,
    "섬진강자전거길": 148,
    "오천자전거길": 105,
    "동해안자전거길(강원구간)": 242,
    "동해안자전거길(경북구간)": 76,
    # 제주
    "제주환상": 234,
}

# ───────────────────────────────────────────────────────────────────────────────
# 1) 최상위 카테고리 매핑
# ───────────────────────────────────────────────────────────────────────────────
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
    "북한강자전거길": "그랜드슬램코스",
    "섬진강자전거길": "그랜드슬램코스",
    "오천자전거길": "그랜드슬램코스",
    "동해안자전거길(강원구간)": "그랜드슬램코스",
    "동해안자전거길(경북구간)": "그랜드슬램코스",
}
TOP_ORDER = ["국토종주코스", "제주환상자전거길", "그랜드슬램코스", "기타코스"]

# ───────────────────────────────────────────────────────────────────────────────
# 2) 기초 유틸
# ───────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def parse_path(s):
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return None

# ───────────────────────────────────────────────────────────────────────────────
# 3) CSV 로더
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_routes(src: str | Path | bytes) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)

    need = {"route", "section", "distance_km"}
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

    df["category"] = df["route"].map(GROUP_MAP).fillna("기타코스")
    df["category"] = pd.Categorical(df["category"], categories=TOP_ORDER, ordered=True)
    return df

@st.cache_data
def load_centers(src: str | Path | bytes) -> pd.DataFrame | None:
    if src is None:
        return None
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)

    need = {"route", "center", "address", "lat", "lng", "id", "seq"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"centers.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    for c in ["category", "route", "center", "address", "id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["lat", "lng", "seq", "leg_km"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["category"] = df["route"].map(GROUP_MAP).fillna("기타코스")
    df["category"] = pd.Categorical(df["category"], categories=TOP_ORDER, ordered=True)
    return df

# ───────────────────────────────────────────────────────────────────────────────
# 4) 데이터 소스 선택
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
# 5) 탭
# ───────────────────────────────────────────────────────────────────────────────
tab = st.radio("", ["🚴 구간(거리) 추적", "📍 인증센터"], horizontal=True, label_visibility="collapsed")

# ───────────────────────────────────────────────────────────────────────────────
# 6) 구간(거리) 추적
# ───────────────────────────────────────────────────────────────────────────────
if tab == "🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")

    # 대분류
    cat_list = ["전체구간"] + [c for c in TOP_ORDER if c in routes["category"].unique()]
    cat = st.sidebar.selectbox("대분류", options=cat_list, index=0)

    df = routes.copy()
    if cat != "전체구간":
        df = df[df["category"] == cat]

    # 노선 멀티셀렉트(기본값: 선택 없음)
    route_names = sorted(df["route"].dropna().unique().tolist())
    route_pick = st.sidebar.multiselect(
        "노선(복수 선택 가능)",
        options=route_names,
        default=[],
        key="route_multi",
        help="표시할 노선을 선택하세요.",
    )
    c1, c2 = st.sidebar.columns(2)
    if c1.button("전체 선택"):
        st.session_state.route_multi = route_names
        route_pick = route_names
    if c2.button("전체 해제"):
        st.session_state.route_multi = []
        route_pick = []

    if not route_pick:
        st.warning("표시할 노선을 선택하세요.")
        st.stop()

    df = df[df["route"].isin(route_pick)].copy()

    # ── 인증센터 기반 경로/거리 생성(있을 때만)
    def centers_polyline_and_km(route_name: str):
        if centers is None:
            return None, np.nan
        g = centers[(centers["route"] == route_name)].dropna(subset=["lat", "lng"]).sort_values("seq")
        if g.empty:
            return None, np.nan
        pts = g[["lng", "lat"]].to_numpy(dtype=float)
        path = pts.tolist()
        if "leg_km" in g.columns and g["leg_km"].notna().any():
            km = float(g["leg_km"].fillna(0).sum())
        else:
            km = 0.0
            for i in range(len(pts) - 1):
                km += haversine_km(pts[i][1], pts[i][0], pts[i + 1][1], pts[i + 1][0])
        return path, km

    by_route = {}
    for rname in df["route"].unique():
        p, k = centers_polyline_and_km(rname)
        by_route[rname] = {"derived_path": p, "derived_km": k}

    df["__derived_km"] = df["route"].map(lambda r: by_route.get(r, {}).get("derived_km", np.nan))
    df["__derived_path"] = df["route"].map(lambda r: by_route.get(r, {}).get("derived_path", None))

    # 표시/계산에 쓸 km (우선순위: routes.distance_km > 공식거리 > centers 파생거리)
    df["__display_km"] = np.where(
        df["distance_km"].notna() & (df["distance_km"] > 0),
        df["distance_km"],
        np.where(
            df["route"].map(OFFICIAL_TOTALS).notna(),
            df["route"].map(OFFICIAL_TOTALS).astype(float),
            df["__derived_km"],
        ),
    )

    # 요약(선택 노선 공식 총거리)
    with st.expander("선택 노선 총거리 요약", expanded=True):
        summary = pd.DataFrame(
            {"route": route_pick, "총거리(km)": [float(OFFICIAL_TOTALS.get(r, df[df["route"] == r]["__display_km"].sum())) for r in route_pick]}
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # 완료 체크 상태
    if "done_ids" not in st.session_state:
        st.session_state.done_ids = set()
    df["완료"] = df["id"].isin(st.session_state.done_ids)

    # 표에는 표시용 거리 사용
    df_edit = df[["category", "route", "section", "__display_km", "완료"]].rename(columns={"__display_km": "distance_km"})
    edited = st.data_editor(df_edit, use_container_width=True, hide_index=True, key="editor_routes")

    # 체크 반영
    merge_key = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)
    id_map = dict(zip(merge_key, df["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        k = (str(row["route"]) + "@" + str(row["section"])).replace(" ", "")
        _id = id_map.get(k)
        if _id and bool(row["완료"]):
            new_done.add(_id)
    st.session_state.done_ids = new_done

    # KPI — 표시용 거리 기준
    total_km = float(df["__display_km"].sum())
    done_km = float(df[df["id"].isin(st.session_state.done_ids)]["__display_km"].sum())
    left_km = max(total_km - done_km, 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    if len(route_pick) == 1:
        c4.metric("공식 노선 총거리", f"{float(OFFICIAL_TOTALS.get(route_pick[0], total_km)):,.1f} km")
    else:
        c4.metric("공식 노선 총거리", "다중 선택")

    # 지도: routes.path or centers 파생 경로
    df["__path"] = None
    if "path" in df.columns:
        df["__path"] = df["path"].dropna().map(parse_path)
    df.loc[df["__path"].isna(), "__path"] = df.loc[df["__path"].isna(), "__derived_path"]

    layers = []
    paths_df = df.dropna(subset=["__path"]).copy()
    if not paths_df.empty:
        paths_df["__color"] = paths_df["id"].apply(lambda x: [28, 200, 138] if x in st.session_state.done_ids else [230, 57, 70])
        layers.append(
            pdk.Layer(
                "PathLayer",
                paths_df.rename(columns={"__path": "path"}),
                get_path="path",
                get_color="__color",
                width_scale=3,
                width_min_pixels=3,
                pickable=True,
            )
        )

    # 센터 마커(있으면) — 각 행에 색상 배열 부여
    if centers is not None:
        g = centers[centers["route"].isin(route_pick)].dropna(subset=["lat", "lng"]).copy()
        if not g.empty:
            g["__color"] = [[200, 200, 200]] * len(g)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    g.rename(columns={"lat": "latitude", "lng": "longitude"}),
                    get_position='[longitude, latitude]',
                    get_fill_color="__color",
                    get_radius=120,
                    pickable=True,
                )
            )

    # 뷰포인트
    if centers is not None and not centers[centers["route"].isin(route_pick)].dropna(subset=["lat", "lng"]).empty:
        geo = centers[centers["route"].isin(route_pick)].dropna(subset=["lat", "lng"])
        center_lat, center_lng = float(geo["lat"].mean()), float(geo["lng"].mean())
    else:
        center_lat, center_lng = 36.2, 127.5

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7),
            tooltip={"text": "{route} / {section}"},
        ),
        use_container_width=True,
    )

# ───────────────────────────────────────────────────────────────────────────────
# 7) 인증센터(누적거리: 센터 간 segment 합산)
# ───────────────────────────────────────────────────────────────────────────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다.")
        st.stop()

    st.sidebar.header("인증센터 필터")

    centers["category"] = centers["route"].map(GROUP_MAP).fillna("기타코스")
    centers["category"] = pd.Categorical(centers["category"], categories=TOP_ORDER, ordered=True)

    cat_list = ["전체"] + [c for c in TOP_ORDER if c in centers["category"].unique()]
    cat = st.sidebar.selectbox("대분류", cat_list, index=0)

    dfc = centers.copy()
    if cat != "전체":
        dfc = dfc[dfc["category"] == cat]

    # 노선 멀티셀렉트(기본값: 선택 없음)
    route_names = sorted(dfc["route"].dropna().unique().tolist())
    route_pick = st.sidebar.multiselect(
        "노선(복수 선택 가능)",
        options=route_names,
        default=[],
        key="center_route_multi",
        help="인증센터를 확인할 노선을 선택하세요.",
    )
    c1, c2 = st.sidebar.columns(2)
    if c1.button("전체 선택", key="center_all"):
        st.session_state.center_route_multi = route_names
        route_pick = route_names
    if c2.button("전체 해제", key="center_none"):
        st.session_state.center_route_multi = []
        route_pick = []

    if not route_pick:
        st.warning("노선을 선택하세요.")
        st.stop()

    dfc = dfc[dfc["route"].isin(route_pick)].copy()

    if "done_center_ids" not in st.session_state:
        st.session_state.done_center_ids = set()

    dfc = dfc.sort_values(["route", "seq", "center"], na_position="last").reset_index(drop=True)
    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)

    with st.expander("인증센터 체크(간단 편집)", expanded=True):
        edited = st.data_editor(
            dfc[["category", "route", "seq", "center", "address", "완료"]],
            use_container_width=True,
            hide_index=True,
            key="editor_centers",
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

    # 세그먼트(센터 i → i+1)
    seg_rows = []
    for route, g in dfc.groupby("route"):
        g = g.sort_values("seq")
        recs = g.to_dict("records")
        for i in range(len(recs) - 1):
            a, b = recs[i], recs[i + 1]
            if not pd.isna(a.get("leg_km")):
                dist = float(a["leg_km"])
            else:
                dist = haversine_km(a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng"))
                if pd.isna(dist):
                    dist = 0.0
            seg_rows.append(
                {
                    "route": route,
                    "start_center": a["center"],
                    "end_center": b["center"],
                    "start_lat": a.get("lat"),
                    "start_lng": a.get("lng"),
                    "end_lat": b.get("lat"),
                    "end_lng": b.get("lng"),
                    "distance_km": dist,
                    "done": bool(a["완료"] and b["완료"]),
                }
            )
    seg_df = pd.DataFrame(seg_rows)

    # KPI
    if not seg_df.empty:
        total_km_centers = float(seg_df["distance_km"].sum())
        done_km_centers = float(seg_df.loc[seg_df["done"], "distance_km"].sum())
        left_km_centers = max(total_km_centers - done_km_centers, 0.0)
    else:
        total_km_centers = done_km_centers = left_km_centers = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 인증센터 수", f"{dfc.shape[0]:,}")
    c2.metric("완료한 인증센터", f"{int(dfc[dfc['완료']].shape[0]):,}")
    c3.metric("센터 기준 누적거리", f"{done_km_centers:,.1f} km")
    c4.metric("센터 기준 남은 거리", f"{left_km_centers:,.1f} km")

    # 지도(경로 + 마커)
    layers = []
    if not seg_df.empty:
        for flag, color in [(True, [28, 200, 138]), (False, [230, 57, 70])]:
            src = seg_df[seg_df["done"] == flag].copy()
            if src.empty:
                continue
            src["__path"] = src.apply(
                lambda r: [[r["start_lng"], r["start_lat"]], [r["end_lng"], r["end_lat"]]], axis=1
            )
            src["__color"] = [color] * len(src)
            layers.append(
                pdk.Layer(
                    "PathLayer",
                    src,
                    get_path="__path",
                    get_color="__color",
                    width_scale=3,
                    width_min_pixels=3,
                    pickable=True,
                )
            )

    geo = dfc.dropna(subset=["lat", "lng"]).copy()
    if not geo.empty:
        geo["__color"] = geo["완료"].map(lambda b: [28, 200, 138] if b else [230, 57, 70])
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                geo.rename(columns={"lat": "latitude", "lng": "longitude"}),
                get_position='[longitude, latitude]',
                get_fill_color="__color",
                get_radius=160,
                pickable=True,
            )
        )
        vlat, vlng = float(geo["latitude"].mean()), float(geo["longitude"].mean())
    else:
        vlat, vlng = 36.2, 127.5

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=7),
            tooltip={"text": "{route}\n{start_center} → {end_center}"},
        ),
        use_container_width=True,
    )
