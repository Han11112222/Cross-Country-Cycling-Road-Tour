# app.py — Country Cycling Route Tracker (공식 총거리 + 인증센터 경로 자동선)
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# ───────────────────────────────────────────────────────────────────────────────
# 0) 공식 총거리(국토부/자전거행복나눔 고정값)
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
    "낙동강자전거길": 389,  # 공식 총거리
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

    for c in ["category", "route", "center", "address", "id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # seq(인증센터 순번) 없으면 None으로 두고, 나중에 이름 기준 정렬
    if "seq" in df.columns:
        df["seq"] = pd.to_numeric(df["seq"], errors="coerce")

    for c in ["lat", "lng"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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
# 4) 구간(거리) 추적
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

    def official_total(route: str) -> float:
        return float(OFFICIAL_TOTALS.get(route, routes.loc[routes["route"] == route, "distance_km"].sum()))

    with st.expander("선택 노선 총거리 요약", expanded=False):
        summary = pd.DataFrame({
            "route": route_pick,
            "총거리(km)": [official_total(r) for r in route_pick]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    if "done_ids" not in st.session_state:
        st.session_state.done_ids = set()
    df["완료"] = df["id"].isin(st.session_state.done_ids)

    edited = st.data_editor(
        df[["category", "route", "section", "distance_km", "완료"]],
        use_container_width=True, hide_index=True, key="editor_routes",
    )

    merge_key = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)
    id_map = dict(zip(merge_key, df["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        k = (str(row["route"]) + "@" + str(row["section"])).replace(" ", "")
        _id = id_map.get(k)
        if _id and bool(row["완료"]):
            new_done.add(_id)
    st.session_state.done_ids = new_done

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

    # 지도 데이터 구성
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

    layers = []

    # ① routes.csv의 path가 있으면 PathLayer
    routes_with_path = df[df["__path"].notna()].copy()
    if not routes_with_path.empty:
        routes_with_path["__color"] = routes_with_path["id"].apply(
            lambda x: [28, 200, 138] if x in st.session_state.done_ids else [230, 57, 70]
        )
        layers.append(pdk.Layer(
            "PathLayer",
            routes_with_path, get_path="__path", get_color="__color",
            width_scale=3, width_min_pixels=3, pickable=True
        ))

    # ② 인증센터 기반 폴리라인(경로 fallback) — centers.csv 좌표가 있을 때
    if centers is not None:
        cg = centers.dropna(subset=["lat", "lng"])
        if not cg.empty:
            for r in route_pick:
                g = cg[cg["route"] == r].copy()
                if g.empty:
                    continue
                if "seq" in g.columns and g["seq"].notna().any():
                    g = g.sort_values("seq")
                else:
                    g = g.sort_values("center")
                coords = g.apply(lambda s: [float(s["lng"]), float(s["lat"])], axis=1).tolist()
                if len(coords) >= 2:
                    poly = pd.DataFrame({"route":[r], "__path":[coords]})
                    poly["__color"] = [[66, 135, 245]]  # 파란 선
                    layers.append(pdk.Layer(
                        "PathLayer",
                        poly, get_path="__path", get_color="__color",
                        width_scale=2, width_min_pixels=2, pickable=False
                    ))
            # 인증센터 마커
            cg["longitude"] = cg["lng"].astype(float)
            cg["latitude"] = cg["lat"].astype(float)
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                cg, get_position='[longitude, latitude]',
                get_fill_color=[66,135,245], get_radius=150, pickable=True
            ))

    # ③ 시작/끝 점(좌표가 있을 때만)
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
    if pts:
        pts_df = pd.DataFrame(pts)
        pts_df["__color"] = pts_df["done"].map(lambda b: [28,200,138] if b else [230,57,70])
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            pts_df, get_position='[lng, lat]', get_fill_color='__color',
            get_radius=150, pickable=True
        ))
        center_lng, center_lat = float(pts_df["lng"].mean()), float(pts_df["lat"].mean())
    else:
        # 인증센터 좌표라도 있으면 그걸로 중심 잡기
        if centers is not None and not centers.dropna(subset=["lat","lng"]).empty:
            _g = centers.dropna(subset=["lat","lng"])
            center_lng, center_lat = float(_g["lng"].mean()), float(_g["lat"].mean())
        else:
            center_lng, center_lat = 127.5, 36.2

    view = pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7)
    deck = pdk.Deck(layers=layers, initial_view_state=view,
                    tooltip={"text": "{route}{name}\n{center}\n{address}"})
    st.pydeck_chart(deck, use_container_width=True)

    st.caption("💡 경로를 선으로 보려면 (1) routes.csv의 path 열에 [[lng,lat], ...] JSON을 넣거나 "
               "(2) centers.csv에 각 인증센터의 lat/lng를 채워두면 인증센터를 연결해 선으로 그려드립니다.")

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
    dfc = dfc[dfc["route"].isin(route_pick)].copy()

    if "done_center_ids" not in st.session_state:
        st.session_state.done_center_ids = set()

    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)
    edited = st.data_editor(
        dfc[["category", "route", "center", "address", "완료"]],
        use_container_width=True, hide_index=True, key="editor_centers",
    )
    id_map = dict(zip(dfc["center"], dfc["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        _id = id_map.get(str(row["center"]))
        if _id and bool(row["완료"]):
            new_done.add(_id)
    st.session_state.done_center_ids = new_done

    total_cnt = int(dfc.shape[0])
    done_cnt = int(dfc[dfc["id"].isin(st.session_state.done_center_ids)].shape[0])
    left_cnt = total_cnt - done_cnt
    c1, c2, c3 = st.columns(3)
    c1.metric("선택 인증센터 수", f"{total_cnt:,}")
    c2.metric("완료한 인증센터", f"{done_cnt:,}")
    c3.metric("남은 인증센터", f"{left_cnt:,}")

    geo = dfc.dropna(subset=["lat", "lng"]).copy()
    if not geo.empty:
        geo["__color"] = geo["id"].apply(lambda x: [28, 200, 138] if x in st.session_state.done_center_ids else [230, 57, 70])
        view = pdk.ViewState(latitude=float(geo["lat"].mean()), longitude=float(geo["lng"].mean()), zoom=7)
        layer = pdk.Layer(
            "ScatterplotLayer",
            geo.rename(columns={"lat":"latitude","lng":"longitude"}),
            get_position='[longitude, latitude]',
            get_fill_color="__color",
            get_radius=180, pickable=True,
        )
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                 tooltip={"text": "{route} / {center}\n{address}"}), use_container_width=True)
    else:
        st.info("이 필터에는 좌표(lat,lng)가 없는 센터가 있습니다. centers.csv 에 좌표를 채우면 지도에 표시됩니다.")
