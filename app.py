# app.py — Country Cycling Route Tracker (국토종주 누적거리·인증센터 트래커)
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="국토종주 트래커", layout="wide")


# -----------------------------
# 공통 유틸
# -----------------------------
def _norm(s: str) -> str:
    """공백/슬래시/괄호 등 제거해서 ID로 쓰기 좋은 문자열로."""
    if s is None:
        return ""
    return (
        str(s)
        .strip()
        .replace(" ", "")
        .replace("\u3000", "")
        .replace("/", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )


def _ensure_id(df: pd.DataFrame, cols: Iterable[str], new_col: str) -> pd.DataFrame:
    if new_col not in df.columns:
        key = df[list(cols)].astype(str).agg("@".join, axis=1)
        df[new_col] = key.map(lambda x: _norm(x))
    return df


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# -----------------------------
# 1) 데이터 로드
# -----------------------------
@st.cache_data
def load_routes(src: str | Path | bytes) -> pd.DataFrame:
    """구간(거리) CSV 로드"""
    df = pd.read_csv(src)

    # 필수 컬럼 체크
    need = {"category", "route", "section", "distance_km"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"routes.csv에 필요한 컬럼이 없습니다: {sorted(miss)}")

    # 문자열/숫자 정리
    for c in ["category", "route", "section", "start", "end"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 숫자/좌표
    for c in ["distance_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # id 없으면 route+section으로 생성
    _ensure_id(df, cols=["route", "section"], new_col="id")

    return df


@st.cache_data
def load_centers(src: str | Path | bytes) -> pd.DataFrame:
    """
    인증센터 CSV 로드
    필수: category, route, center, lat, lng (lat/lng는 비워도 됨; 지도가 안 뜰 뿐)
    선택: address, order, id
    """
    df = pd.read_csv(src)
    need = {"category", "route", "center"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"centers.csv에 필요한 컬럼이 없습니다: {sorted(miss)}")

    for c in ["category", "route", "center", "address"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "order" in df.columns:
        df["order"] = pd.to_numeric(df["order"], errors="coerce")

    # 좌표
    for c in ["lat", "lng"]:
        if c in df.columns:
            df[c] = df[c].map(_to_float)

    # id 없으면 route+center로 생성
    _ensure_id(df, cols=["route", "center"], new_col="id")

    return df


def parse_path(s: str | float | None):
    """CSV path 컬럼(JSON 문자열)을 파싱해서 pydeck PathLayer용 list[list[lng,lat]] 반환"""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    return None


# -----------------------------
# 2) 데이터 소스 선택(Repo/업로드)
# -----------------------------
st.sidebar.header("데이터 불러오기")

mode = st.sidebar.radio("방식", ["Repo 내 파일", "CSV 업로드"], index=0)

# routes
if mode == "Repo 내 파일":
    routes_path = Path("data/routes.csv")
    if not routes_path.exists():
        st.error("Repo에 data/routes.csv 가 없습니다. 먼저 CSV를 추가해주세요.")
        st.stop()
    routes = load_routes(routes_path)
else:
    up_routes = st.sidebar.file_uploader("routes.csv 업로드", type=["csv"], key="routes_up")
    if up_routes is None:
        st.info("routes.csv를 올리면 시작합니다.")
        st.stop()
    routes = load_routes(up_routes)

# centers (있으면 로드)
centers: pd.DataFrame | None = None
try:
    if mode == "Repo 내 파일":
        centers_path = Path("data/centers.csv")
        if centers_path.exists():
            centers = load_centers(centers_path)
    else:
        up_centers = st.sidebar.file_uploader("centers.csv 업로드(선택)", type=["csv"], key="centers_up")
        if up_centers is not None:
            centers = load_centers(up_centers)
except Exception as e:
    st.sidebar.warning(f"centers.csv 로드 오류: {e}")


# -----------------------------
# 3) 진행 상태(구간/인증센터) 저장/불러오기
# -----------------------------
if "route_done_ids" not in st.session_state:
    st.session_state.route_done_ids: set[str] = set()
if "center_done_ids" not in st.session_state:
    st.session_state.center_done_ids: set[str] = set()

with st.sidebar.expander("진행상태 저장/불러오기", expanded=False):
    up_state = st.file_uploader("진행상태 불러오기(.json)", type=["json"], key="state_up")
    if up_state:
        try:
            obj = json.load(up_state)
            if isinstance(obj, dict):
                st.session_state.route_done_ids = set(obj.get("route_done_ids", []))
                st.session_state.center_done_ids = set(obj.get("center_done_ids", []))
            elif isinstance(obj, list):
                # 구버전 호환(구간만 저장했던 경우)
                st.session_state.route_done_ids = set(obj)
            st.success("진행상태를 불러왔습니다.")
        except Exception as e:
            st.error(f"불러오기 실패: {e}")

    state_obj = {
        "route_done_ids": sorted(list(st.session_state.route_done_ids)),
        "center_done_ids": sorted(list(st.session_state.center_done_ids)),
    }
    st.download_button(
        "진행상태 저장(.json)",
        data=json.dumps(state_obj, ensure_ascii=False, indent=2),
        file_name="progress.json",
        mime="application/json",
        use_container_width=True,
    )


# -----------------------------
# 4) 탭: [구간(거리) 추적] / [인증센터]
# -----------------------------
t1, t2 = st.tabs(["🚴 구간(거리) 추적", "📍 인증센터"])

# ===== 탭 1. 구간(거리) 추적 =====
with t1:
    st.subheader("구간(거리) 추적")

    # 필터
    col1, col2 = st.columns([1, 2])
    with col1:
        cat_list = ["전체구간"] + sorted(routes["category"].dropna().unique().tolist())
        cat = st.selectbox("대분류", options=cat_list, index=0, key="route_cat")
    df_r = routes.copy()
    if cat != "전체구간":
        df_r = df_r[df_r["category"] == cat]

    route_names = sorted(df_r["route"].dropna().unique().tolist())
    if not route_names:
        st.warning("선택한 대분류에 노선이 없습니다.")
        st.stop()

    with col2:
        pick_routes = st.multiselect("노선(복수 선택 가능)", options=route_names, default=route_names, key="route_pick")
    if not pick_routes:
        st.info("노선을 최소 1개 이상 선택하세요.")
        st.stop()

    df_r = df_r[df_r["route"].isin(pick_routes)].copy()
    st.caption(f"🔎 필터: 카테고리 **{cat}**, 노선 **{', '.join(pick_routes)}**")

    # 완료 체크
    df_r["완료"] = df_r["id"].isin(st.session_state.route_done_ids)
    edited = st.data_editor(
        df_r[["category", "route", "section", "distance_km", "완료"]],
        use_container_width=True, hide_index=True, key="route_editor"
    )

    # 반영
    merge_key = (df_r["route"].astype(str) + "@" + df_r["section"].astype(str)).map(_norm)
    id_map = dict(zip(merge_key, df_r["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        k = _norm(str(row["route"]) + "@" + str(row["section"]))
        _id = id_map.get(k)
        if _id and bool(row["완료"]):
            new_done.add(_id)
    st.session_state.route_done_ids = new_done

    # KPI
    total_km = float(df_r["distance_km"].sum())
    done_km = float(df_r[df_r["id"].isin(st.session_state.route_done_ids)]["distance_km"].sum())
    left_km = max(total_km - done_km, 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 구간 총거리", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("완료율", f"{(done_km / total_km * 100 if total_km > 0 else 0):.1f}%")

    # 지도 (path 우선, 없으면 점)
    df_r["__path"] = df_r["path"].map(parse_path) if "path" in df_r.columns else None
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
                    "done": bool(r["id"] in st.session_state.route_done_ids),
                })
    pts_df = pd.DataFrame(pts)

    if len(pts_df) > 0:
        center_lng, center_lat = float(pts_df["lng"].mean()), float(pts_df["lat"].mean())
    else:
        center_lng, center_lat = 127.5, 36.2

    layers = []
    if not paths.empty:
        paths["__color"] = paths["id"].apply(
            lambda x: [28, 200, 138] if x in st.session_state.route_done_ids else [230, 57, 70]
        )
        layers.append(
            pdk.Layer(
                "PathLayer",
                paths,
                get_path="__path",
                get_color="__color",
                width_scale=3,
                width_min_pixels=3,
                pickable=True,
            )
        )
    if not pts_df.empty:
        pts_df["__color"] = pts_df["done"].map(lambda b: [28, 200, 138] if b else [230, 57, 70])
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                pts_df,
                get_position='[lng, lat]',
                get_fill_color='__color',
                get_radius=150,
                pickable=True,
            )
        )

    view = pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7)
    deck = pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{name}"})
    st.pydeck_chart(deck, use_container_width=True)

    st.caption("💡 선으로 보려면 routes.csv의 path 열에 [[lng,lat],[lng,lat],...] 형식 JSON을 넣으세요. 없으면 시작/끝 좌표가 마커로 표시됩니다.")


# ===== 탭 2. 인증센터 =====
with t2:
    st.subheader("인증센터")

    if centers is None or centers.empty:
        st.info("centers.csv가 없거나 비었습니다. data/centers.csv를 추가하면 이 탭에서 노선별 인증센터를 볼 수 있습니다.")
    else:
        # 필터 (구간 탭과 동일 UX)
        col1, col2 = st.columns([1, 2])
        with col1:
            cat_list_c = ["전체"] + sorted(centers["category"].dropna().unique().tolist())
            cat_c = st.selectbox("대분류", options=cat_list_c, index=0, key="center_cat")
        df_c = centers.copy()
        if cat_c != "전체":
            df_c = df_c[df_c["category"] == cat_c]

        route_names_c = sorted(df_c["route"].dropna().unique().tolist())
        with col2:
            pick_routes_c = st.multiselect("노선(복수 선택 가능)", options=route_names_c, default=route_names_c, key="center_pick")
        if not pick_routes_c:
            st.info("노선을 최소 1개 이상 선택하세요.")
            st.stop()

        df_c = df_c[df_c["route"].isin(pick_routes_c)].copy()

        # 완료 체크
        df_c["완료"] = df_c["id"].isin(st.session_state.center_done_ids)
        cols_show = ["category", "route", "center", "address"] if "address" in df_c.columns else ["category", "route", "center"]
        edited_c = st.data_editor(
            df_c[cols_show + ["완료"]],
            use_container_width=True, hide_index=True, key="center_editor"
        )

        # 반영
        id_map_c = dict(zip(df_c["id"], df_c["id"]))
        new_done_c = set()
        for _, row in edited_c.iterrows():
            # row에는 id가 없으므로 center+route로 재생성
            rid = _norm(str(row["route"]) + "@" + str(row["center"]))
            _id = id_map_c.get(rid)
            if _id and bool(row["완료"]):
                new_done_c.add(_id)
        st.session_state.center_done_ids = new_done_c

        # KPI
        total_cnt = int(df_c.shape[0])
        done_cnt = int(df_c[df_c["id"].isin(st.session_state.center_done_ids)].shape[0])
        c1, c2 = st.columns(2)
        c1.metric("선택 인증센터 수", f"{total_cnt} 곳")
        c2.metric("완료한 인증센터", f"{done_cnt} 곳")

        # 지도(좌표 있는 것만)
        has_coord = df_c.dropna(subset=["lat", "lng"]).copy()
        if has_coord.empty:
            st.info("centers.csv에 lat,lng를 채우면 지도로 표시됩니다.")
        else:
            has_coord["done"] = has_coord["id"].isin(st.session_state.center_done_ids)
            has_coord["__color"] = has_coord["done"].map(lambda b: [28, 200, 138] if b else [230, 57, 70])
            layer = pdk.Layer(
                "ScatterplotLayer",
                has_coord,
                get_position='[lng, lat]',
                get_fill_color='__color',
                get_radius=120,
                pickable=True,
            )
            view = pdk.ViewState(
                latitude=float(has_coord["lat"].mean()),
                longitude=float(has_coord["lng"].mean()),
                zoom=8,
            )
            st.pydeck_chart(
                pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{route} - {center}\n{address}"}),
                use_container_width=True,
            )

