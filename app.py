# app.py — Country Cycling Route Tracker (구간·거리 + 인증센터)
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="국토종주 누적거리 · 인증센터", layout="wide")


# ======================================
# Data loaders
# ======================================
@st.cache_data
def load_routes(src: Union[str, Path, bytes]) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)

    need = {"category", "route", "section", "distance_km"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"routes.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    # 문자열/숫자 정리
    for c in ["category", "route", "section"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    for c in ["distance_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # id 없으면 route+section으로 생성
    if "id" not in df.columns:
        df["id"] = (
            df["route"].astype(str) + "@" + df["section"].astype(str)
        ).str.replace(r"\s+", "", regex=True)

    return df


@st.cache_data
def load_centers(src: Union[str, Path, bytes]) -> pd.DataFrame:
    """인증센터 목록(가점/좌표 관리)"""
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)
    need = {"category", "route", "center", "address"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"centers.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    for c in ["category", "route", "center", "address"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    for c in ["lat", "lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "id" not in df.columns:
        df["id"] = (
            df["route"].astype(str) + "#" + df["center"].astype(str)
        ).str.replace(r"\s+", "", regex=True)

    return df


# ======================================
# Data entrance (repo / upload)
# ======================================
st.sidebar.header("데이터")
mode = st.sidebar.radio("불러오기 방식", ["Repo 내 파일", "CSV 업로드"], index=0)

if mode == "Repo 내 파일":
    routes_path = Path("data/routes.csv")
    centers_path = Path("data/centers.csv")
    if not routes_path.exists():
        st.error("data/routes.csv 가 없습니다. 먼저 업로드해주세요.")
        st.stop()
    if not centers_path.exists():
        st.error("data/centers.csv 가 없습니다. 먼저 업로드해주세요.")
        st.stop()
    routes = load_routes(routes_path)
    centers = load_centers(centers_path)
else:
    up_routes = st.sidebar.file_uploader("routes.csv 업로드", type=["csv"], key="routes_up")
    up_centers = st.sidebar.file_uploader("centers.csv 업로드", type=["csv"], key="centers_up")
    if not up_routes or not up_centers:
        st.info("routes.csv, centers.csv를 모두 올리면 시작합니다.")
        st.stop()
    routes = load_routes(up_routes)
    centers = load_centers(up_centers)

# 상태 저장 (구간 완료 / 인증센터 도장완료)
if "route_done_ids" not in st.session_state:
    st.session_state.route_done_ids = set()
if "center_done_ids" not in st.session_state:
    st.session_state.center_done_ids = set()

# 진행상태 저장/불러오기
with st.sidebar.expander("진행상태 저장/불러오기", expanded=False):
    up_state = st.file_uploader("불러오기(.json)", type=["json"], key="state_up")
    if up_state:
        try:
            state = json.load(up_state)
            st.session_state.route_done_ids = set(state.get("route_done_ids", []))
            st.session_state.center_done_ids = set(state.get("center_done_ids", []))
            st.success("진행상태를 불러왔습니다.")
        except Exception as e:
            st.error(f"불러오기 실패: {e}")

    export = json.dumps(
        {
            "route_done_ids": sorted(list(st.session_state.route_done_ids)),
            "center_done_ids": sorted(list(st.session_state.center_done_ids)),
        },
        ensure_ascii=False,
    )
    st.download_button(
        "진행상태 저장(.json)",
        data=export,
        file_name="progress.json",
        mime="application/json",
    )

# ======================================
# Tabs
# ======================================
tab_track, tab_centers = st.tabs(["🚴 구간(거리) 추적", "📍 인증센터"])

# ----------------------------
# 🚴 구간(거리) 추적
# ----------------------------
with tab_track:
    st.subheader("구간(거리) 추적")

    # 필터
    cat_list = ["전체구간"] + sorted(routes["category"].dropna().unique().tolist())
    cat_pick = st.selectbox("대분류", cat_list, index=0)
    df_r = routes.copy()
    if cat_pick != "전체구간":
        df_r = df_r[df_r["category"] == cat_pick]

    route_names = sorted(df_r["route"].dropna().unique().tolist())
    pick_routes = st.multiselect("노선(복수 선택 가능)", route_names, default=route_names)
    if not pick_routes:
        st.warning("노선을 1개 이상 선택하세요.")
        st.stop()
    df_r = df_r[df_r["route"].isin(pick_routes)].copy()

    # 편집 표(완료 체크)
    df_r["완료"] = df_r["id"].isin(st.session_state.route_done_ids)
    edited = st.data_editor(
        df_r[["category", "route", "section", "distance_km", "완료"]],
        use_container_width=True, hide_index=True, key="route_editor"
    )
    # 완료 반영
    key_series = (df_r["route"].astype(str) + "@" + df_r["section"].astype(str)).str.replace(r"\s+", "", regex=True)
    id_map = dict(zip(key_series, df_r["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        k = (str(row["route"]) + "@" + str(row["section"])).replace(" ", "")
        rid = id_map.get(k)
        if rid and bool(row["완료"]):
            new_done.add(rid)
    st.session_state.route_done_ids = new_done

    # KPI (선택집합 총거리/완료/잔여)
    total_km = float(df_r["distance_km"].sum())
    done_km = float(df_r[df_r["id"].isin(st.session_state.route_done_ids)]["distance_km"].sum())
    left_km = max(total_km - done_km, 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 구간 총거리", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("완료율", f"{(done_km/total_km*100 if total_km>0 else 0):.1f}%")

    # ▶ 노선별 전체 길이(정확한 '코스 총합' 확인용)
    st.markdown("#### 노선별 전체 길이")
    st.dataframe(
        df_r.groupby("route", as_index=False)["distance_km"].sum().rename(columns={"distance_km":"총 거리(km)"}),
        use_container_width=True,
        hide_index=True
    )

    # 지도 (path가 있으면 선, 없으면 시작/끝 점)
    def parse_path(s):
        try:
            val = json.loads(s)
            if isinstance(val, list):
                return val
        except Exception:
            pass
        return None

    df_r["__path"] = None
    if "path" in df_r.columns:
        df_r.loc[df_r["path"].notna(), "__path"] = df_r.loc[df_r["path"].notna(), "path"].map(parse_path)

    paths = df_r[df_r["__path"].notna()].copy()
    pts = []
    for _, r in df_r.iterrows():
        for (lng, lat, label) in [
            (r.get("start_lng"), r.get("start_lat"), "start"),
            (r.get("end_lng"), r.get("end_lat"), "end"),
        ]:
            if pd.notna(lng) and pd.notna(lat):
                pts.append({
                    "lng": float(lng), "lat": float(lat),
                    "name": f"{r['route']} / {r['section']} ({label})",
                    "done": bool(r["id"] in st.session_state.route_done_ids)
                })
    pts_df = pd.DataFrame(pts)

    if not pts_df.empty:
        center_lng, center_lat = float(pts_df["lng"].mean()), float(pts_df["lat"].mean())
    else:
        center_lng, center_lat = 127.5, 36.2

    layers = []
    if not paths.empty:
        paths["__color"] = paths["id"].apply(lambda x: [28,200,138] if x in st.session_state.route_done_ids else [230,57,70])
        layers.append(
            pdk.Layer(
                "PathLayer", paths, get_path="__path", get_color="__color",
                width_scale=3, width_min_pixels=3, pickable=True
            )
        )
    if not pts_df.empty:
        pts_df["__color"] = pts_df["done"].map(lambda b: [28,200,138] if b else [230,57,70])
        layers.append(
            pdk.Layer(
                "ScatterplotLayer", pts_df, get_position='[lng, lat]',
                get_fill_color='__color', get_radius=150, pickable=True
            )
        )

    view = pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7)
    deck = pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text":"{name}"})
    st.pydeck_chart(deck, use_container_width=True)

    st.caption("💡 선으로 보이게 하려면 routes.csv의 **path** 열에 `[ [lng,lat], [lng,lat], ... ]` 형식 JSON을 넣으세요.")


# ----------------------------
# 📍 인증센터
# ----------------------------
with tab_centers:
    st.subheader("인증센터")

    # 필터
    cat_c_list = ["전체"] + sorted(centers["category"].dropna().unique().tolist())
    cat_c = st.selectbox("대분류", options=cat_c_list, index=0, key="c_cat")
    df_c = centers.copy()
    if cat_c != "전체":
        df_c = df_c[df_c["category"] == cat_c]

    routes_c = sorted(df_c["route"].dropna().unique().tolist())
    pick_c_routes = st.multiselect("노선(복수 선택 가능)", options=routes_c, default=routes_c, key="c_routes")
    if not pick_c_routes:
        st.warning("노선을 1개 이상 선택하세요.")
        st.stop()
    df_c = df_c[df_c["route"].isin(pick_c_routes)].copy()

    # 완료체크
    df_c["완료"] = df_c["id"].isin(st.session_state.center_done_ids)
    edited_c = st.data_editor(
        df_c[["category", "route", "center", "address", "완료"]],
        use_container_width=True, hide_index=True, key="center_editor"
    )
    # 반영
    id_map_c = dict(zip(df_c["center"], df_c["id"]))
    new_center_done = set()
    for _, row in edited_c.iterrows():
        cid = id_map_c.get(str(row["center"]))
        if cid and bool(row["완료"]):
            new_center_done.add(cid)
    st.session_state.center_done_ids = new_center_done

    # KPI
    total_cnt = int(df_c.shape[0])
    done_cnt = int(df_c[df_c["id"].isin(st.session_state.center_done_ids)].shape[0])
    left_cnt = max(total_cnt - done_cnt, 0)
    c1, c2, c3 = st.columns(3)
    c1.metric("선택 인증센터 수", f"{total_cnt:,}")
    c2.metric("완료 인증센터 수", f"{done_cnt:,}")
    c3.metric("남은 인증센터 수", f"{left_cnt:,}")

    # 지도 (좌표 있는 센터만)
    df_c_map = df_c[df_c["lat"].notna() & df_c["lng"].notna()].copy()
    if not df_c_map.empty:
        df_c_map["__color"] = df_c_map["id"].apply(lambda x: [28,200,138] if x in st.session_state.center_done_ids else [230,57,70])
        center_lng, center_lat = float(df_c_map["lng"].mean()), float(df_c_map["lat"].mean())
        layer = pdk.Layer(
            "ScatterplotLayer",
            df_c_map.rename(columns={"lng":"lon"}),
            get_position='[lon, lat]',
            get_fill_color="__color",
            get_radius=200,
            pickable=True,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=8),
            tooltip={"text": "{route} · {center}\n{address}"}
        )
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.info("선택한 항목에 좌표(lat,lng)가 있는 인증센터가 없습니다. centers.csv에 좌표를 채워주세요.")
