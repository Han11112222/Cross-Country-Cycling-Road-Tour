# app.py — Country Cycling Route Tracker (국토종주 누적거리 + 인증센터)
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# --------------------------------------------------
# CSV 로더
# --------------------------------------------------
@st.cache_data
def load_routes(src: str | Path | bytes) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)

    need = {"category", "route", "section"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"routes.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    # 기본 컬럼들 정리
    for c in ["category", "route", "section", "id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 숫자형
    for c in ["distance_km", "route_total_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # id 없으면 route+section으로 생성
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
    for c in ["lat", "lng"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # id 없으면 route+center로 생성
    if "id" not in df.columns or df["id"].isna().any():
        df["id"] = np.where(
            df.get("id").isna() if "id" in df.columns else True,
            (df["route"] + "@" + df["center"]).str.replace(r"\s+", "", regex=True),
            df.get("id", "")
        )
    return df


# --------------------------------------------------
# 데이터 선택(Repo/업로드)
# --------------------------------------------------
st.sidebar.header("데이터")
mode = st.sidebar.radio("불러오기 방식", ["Repo 내 파일", "CSV 업로드"], index=0)

if mode == "Repo 내 파일":
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

# --------------------------------------------------
# 탭
# --------------------------------------------------
tab = st.radio("", ["🚴 구간(거리) 추적", "📍 인증센터"], horizontal=True, label_visibility="collapsed")

# 공통: 진행상태 보관
if "done_ids" not in st.session_state:
    st.session_state.done_ids = set()
if "done_center_ids" not in st.session_state:
    st.session_state.done_center_ids = set()

# --------------------------------------------------
# ① 구간(거리) 추적
# --------------------------------------------------
if tab == "🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")
    cat_list = ["전체구간"] + sorted(routes["category"].dropna().unique().tolist())
    cat = st.sidebar.selectbox("대분류", cat_list, index=0)

    df = routes.copy()
    if cat != "전체구간":
        df = df[df["category"] == cat]

    route_names = sorted(df["route"].dropna().unique().tolist())
    route_pick = st.sidebar.multiselect("노선(복수 선택 가능)", route_names, default=route_names)
    if not route_pick:
        st.stop()
    df = df[df["route"].isin(route_pick)].copy()

    # ── 선택 노선 총거리 요약 (route_total_km 우선, 없으면 distance_km 합계)
    def total_by_route(sub: pd.DataFrame) -> float:
        if "route_total_km" in sub.columns and sub["route_total_km"].notna().any():
            return float(sub["route_total_km"].dropna().iloc[0])
        return float(sub["distance_km"].fillna(0).sum())

    with st.expander("선택 노선 총거리 요약", expanded=True):
        rows = []
        for r in route_pick:
            sub = routes[routes["route"] == r]
            rows.append({"route": r, "총거리(km)": total_by_route(sub)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 진행 체크 박스 테이블
    df["완료"] = df["id"].isin(st.session_state.done_ids)
    edited = st.data_editor(
        df[["category", "route", "section", "distance_km", "완료"]],
        use_container_width=True, hide_index=True, key="editor_routes"
    )

    # 에디터 결과 반영
    merge_key = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)
    id_map = dict(zip(merge_key, df["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        k = (str(row["route"]) + "@" + str(row["section"])).replace(" ", "")
        _id = id_map.get(k)
        if _id and bool(row["완료"]):
            new_done.add(_id)
    st.session_state.done_ids = new_done

    # KPI 계산 로직
    # 1) 구간별 distance_km 가 있으면 그 합으로 진행거리 계산
    total_km = 0.0
    done_km  = 0.0
    for r in route_pick:
        sub = df[df["route"] == r]
        # 총거리
        route_total = total_by_route(routes[routes["route"] == r])
        total_km += route_total

        # 완료거리: 우선 구간합
        seg_total = float(sub["distance_km"].fillna(0).sum())
        seg_done  = float(sub[sub["id"].isin(st.session_state.done_ids)]["distance_km"].fillna(0).sum())

        if seg_total > 0:
            # 구간 거리 제공되는 노선
            done_km += seg_done
        else:
            # 구간 거리가 없으면, 인증센터 비율로 추정(centers.csv 있을 때)
            if centers is not None:
                cc_total = centers[centers["route"] == r]
                if not cc_total.empty:
                    done_cnt = cc_total[cc_total["id"].isin(st.session_state.done_center_ids)].shape[0]
                    ratio = (done_cnt / len(cc_total)) if len(cc_total) else 0
                    done_km += route_total * ratio
            # centers 없으면 0 유지

    left_km  = max(total_km - done_km, 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 노선 총거리", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("완료율", f"{(done_km/total_km*100 if total_km>0 else 0):.1f}%")

    # 지도 (path → 선, 좌표 → 점)
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
                pts.append({
                    "lng": float(lng), "lat": float(lat),
                    "name": f"{r['route']} / {r['section']} ({label})",
                    "done": bool(r["id"] in st.session_state.done_ids)
                })
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

    st.pydeck_chart(pdk.Deck(layers=layers,
                             initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7),
                             tooltip={"text": "{name}"}),
                    use_container_width=True)

    st.caption("💡 *routes.csv* 의 `route_total_km` 이 있으면 그 값을 총거리로 사용하고, "
               "`distance_km` 이 없는 노선은 인증센터 완료 비율로 진행거리를 추정합니다.")

# --------------------------------------------------
# ② 인증센터
# --------------------------------------------------
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

    # KPI
    total_cnt = int(dfc.shape[0])
    done_cnt  = int(dfc[dfc["id"].isin(st.session_state.done_center_ids)].shape[0])
    left_cnt  = total_cnt - done_cnt
    c1, c2, c3 = st.columns(3)
    c1.metric("선택 인증센터 수", f"{total_cnt:,}")
    c2.metric("완료한 인증센터", f"{done_cnt:,}")
    c3.metric("남은 인증센터", f"{left_cnt:,}")

    # 지도: 좌표가 있는 센터만
    geo = dfc.dropna(subset=["lat", "lng"]).copy()
    geo["__color"] = geo["id"].apply(lambda x: [28,200,138] if x in st.session_state.done_center_ids else [230,57,70])
    if not geo.empty:
        layer = pdk.Layer(
            "ScatterplotLayer",
            geo.rename(columns={"lat": "latitude", "lng": "longitude"}),
            get_position='[longitude, latitude]',
            get_fill_color="__color",
            get_radius=180,
            pickable=True,
        )
        view = pdk.ViewState(latitude=float(geo["latitude"].mean()),
                             longitude=float(geo["longitude"].mean()), zoom=7)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                 tooltip={"text": "{route} / {center}\n{address}"}),
                        use_container_width=True)
    else:
        st.info("이 필터에는 좌표(lat,lng)가 있는 인증센터가 없습니다. "
                "centers.csv 에 좌표를 채워 넣으면 지도에 표시됩니다.")
