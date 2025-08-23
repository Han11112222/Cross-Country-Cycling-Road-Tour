# app.py — 누적거리 제대로 계산 + 인증센터 좌표/대체거리 + 상위 카테고리 재정의
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# ───────────────────────────────────────────────────────────────────────────────
# 0) 공식 총거리(자전거행복나눔 기준) — 노선 총거리 표시에 사용
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
# 1) 최상위 카테고리 재정의
#    - CSV의 category 값을 무시하고, 아래 규칙으로 덮어씁니다.
# ───────────────────────────────────────────────────────────────────────────────
GROUP_MAP = {
    # 4대강코스
    "낙동강자전거길": "4대강코스",
    "금강자전거길": "4대강코스",
    "영산강자전거길": "4대강코스",
    "한강종주자전거길(서울구간)": "4대강코스",  # 한강계열

    # 국토종주코스(인천~부산 주행 구성요소를 이 그룹으로)
    "아라자전거길": "국토종주코스",
    "남한강자전거길": "국토종주코스",
    "새재자전거길": "국토종주코스",
    # '낙동강자전거길'은 위에서 4대강으로 분류하지만, 국토종주 구성에도 포함됨
    # 필요시 여기로 이동해도 됨.

    # 기타/연결·내륙
    "동해안자전거길(강원구간)": "기타코스",
    "동해안자전거길(경북구간)": "기타코스",
    "섬진강자전거길": "기타코스",
    "오천자전거길": "기타코스",
    "북한강자전거길": "기타코스",
    "제주환상": "기타코스",
}

TOP_ORDER = ["그랜드슬램코스", "국토종주코스", "4대강코스", "기타코스"]  # 노출 순서용

# ───────────────────────────────────────────────────────────────────────────────
# 2) 인증센터 좌표 기본값(낙동강 11개) — centers.csv에 좌표가 없으면 자동 보정
#    ※ ±1~2km 근사치. 필요시 파일에서 더 정확히 덮어써도 됩니다.
# ───────────────────────────────────────────────────────────────────────────────
DEFAULT_CENTER_COORDS = {
    "NAK-01": (36.4410, 128.2160),  # 상주 상풍교
    "NAK-02": (36.4140, 128.2620),  # 상주보
    "NAK-03": (36.4200, 128.4050),  # 낙단보
    "NAK-04": (36.1450, 128.3550),  # 구미보
    "NAK-05": (36.0200, 128.3890),  # 칠곡보
    "NAK-06": (35.8150, 128.4600),  # 강정고령보(ARC)
    "NAK-07": (35.6930, 128.4300),  # 달성보
    "NAK-08": (35.5100, 128.4300),  # 합천창녕보
    "NAK-09": (35.3850, 128.5050),  # 창녕함안보
    "NAK-10": (35.3770, 129.0140),  # 양산 물문화관
    "NAK-11": (35.0960, 128.9650),  # 낙동강 하굿둑
}

# ───────────────────────────────────────────────────────────────────────────────
# 3) 유틸
# ───────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])): return np.nan
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ───────────────────────────────────────────────────────────────────────────────
# 4) CSV 로더
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

    # id 없으면 생성
    if "id" not in df.columns:
        df["id"] = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)

    # ⛳ CSV category 무시하고 최상위 그룹으로 덮어쓰기
    df["category"] = df["route"].map(GROUP_MAP).fillna("기타코스")
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
    for c in ["lat", "lng", "seq", "leg_km"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # id 없으면 생성
    if "id" not in df.columns or df["id"].isna().any():
        df["id"] = np.where(
            df.get("id").isna() if "id" in df.columns else True,
            (df["route"] + "@" + df["center"]).str.replace(r"\s+", "", regex=True),
            df.get("id", "")
        )

    # 좌표 자동 보정(낙동강 기본 좌표)
    for i, r in df.iterrows():
        _id = r.get("id")
        if pd.isna(r.get("lat")) or pd.isna(r.get("lng")):
            if _id in DEFAULT_CENTER_COORDS:
                df.loc[i, "lat"] = DEFAULT_CENTER_COORDS[_id][0]
                df.loc[i, "lng"] = DEFAULT_CENTER_COORDS[_id][1]

    return df


# ───────────────────────────────────────────────────────────────────────────────
# 5) 데이터 불러오기(Repo/업로드)
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
# 6) 탭
# ───────────────────────────────────────────────────────────────────────────────
tab = st.radio("", ["🚴 구간(거리) 추적", "📍 인증센터"], horizontal=True, label_visibility="collapsed")

# ───────────────────────────────────────────────────────────────────────────────
# 7) 구간(거리) 추적
# ───────────────────────────────────────────────────────────────────────────────
if tab == "🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")
    # 새 상위 카테고리 순서로 노출
    cat_list = ["전체구간"] + [c for c in TOP_ORDER if c in routes["category"].unique()]
    cat = st.sidebar.selectbox("대분류", options=cat_list, index=0)

    df = routes.copy()
    if cat != "전체구간":
        df = df[df["category"] == cat]

    route_names = sorted(df["route"].dropna().unique().tolist())
    route_pick = st.sidebar.multiselect("노선(복수 선택 가능)", route_names, default=route_names)
    if not route_pick:
        st.stop()
    df = df[df["route"].isin(route_pick)].copy()

    # 공식 총거리 요약
    def official_total(route: str) -> float:
        return float(OFFICIAL_TOTALS.get(route, float(routes.loc[routes["route"] == route, "distance_km"].sum())))

    with st.expander("선택 노선 총거리 요약", expanded=True):
        summary = pd.DataFrame({"route": route_pick, "총거리(km)": [official_total(r) for r in route_pick]})
        st.dataframe(summary, use_container_width=True, hide_index=True)

    # 완료체크
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
        if _id and bool(row["완료"]): new_done.add(_id)
    st.session_state.done_ids = new_done

    # KPI(표 합계 기준)
    total_km = float(df["distance_km"].sum())
    done_km  = float(df[df["id"].isin(st.session_state.done_ids)]["distance_km"].sum())
    left_km  = max(total_km - done_km, 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("공식 노선 총거리", "다중 선택" if len(route_pick)!=1 else f"{official_total(route_pick[0]):,.1f} km")

    # 지도(path 또는 시작/끝 좌표)
    def parse_path(s):
        try:
            val = json.loads(s)
            if isinstance(val, list): return val
        except Exception: pass
        return None

    df["__path"] = None
    if "path" in df.columns: df["__path"] = df["path"].dropna().map(parse_path)
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
    if len(pts_df)>0: center_lng, center_lat = float(pts_df["lng"].mean()), float(pts_df["lat"].mean())
    else: center_lng, center_lat = 127.5, 36.2

    layers = []
    if not paths.empty:
        paths["__color"] = paths["id"].apply(lambda x: [28,200,138] if x in st.session_state.done_ids else [230,57,70])
        layers.append(pdk.Layer("PathLayer", paths, get_path="__path", get_color="__color",
                                width_scale=3, width_min_pixels=3, pickable=True))
    if not pts_df.empty:
        pts_df["__color"] = pts_df["done"].map(lambda b: [28,200,138] if b else [230,57,70])
        layers.append(pdk.Layer("ScatterplotLayer", pts_df, get_position='[lng, lat]',
                                get_fill_color='__color', get_radius=150, pickable=True))

    st.pydeck_chart(pdk.Deck(layers=layers,
                             initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7),
                             tooltip={"text": "{name}"}), use_container_width=True)

# ───────────────────────────────────────────────────────────────────────────────
# 8) 인증센터 — 체크하면 누적거리 합산(좌표 없으면 자동좌표/leg_km 대체)
# ───────────────────────────────────────────────────────────────────────────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다.")
        st.stop()

    st.sidebar.header("인증센터 필터")
    # 새 상위 카테고리 체계 노출
    centers["category"] = centers["route"].map(GROUP_MAP).fillna("기타코스")
    cat_list = ["전체"] + [c for c in TOP_ORDER if c in centers["category"].unique()]
    cat = st.sidebar.selectbox("대분류", cat_list, index=0)

    dfc = centers.copy()
    if cat != "전체": dfc = dfc[dfc["category"] == cat]

    route_names = sorted(dfc["route"].dropna().unique().tolist())
    route_pick = st.sidebar.multiselect("노선(복수 선택 가능)", route_names, default=route_names)
    if not route_pick: st.stop()
    dfc = dfc[dfc["route"].isin(route_pick)].copy()

    if "done_center_ids" not in st.session_state:
        st.session_state.done_center_ids = set()

    dfc = dfc.sort_values(["route","seq","center"], na_position="last").reset_index(drop=True)
    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)

    with st.expander("인증센터 체크(간단 편집)", expanded=True):
        edited = st.data_editor(
            dfc[["category","route","seq","center","address","완료"]],
            use_container_width=True, hide_index=True, key="editor_centers",
        )

    # 반영
    id_map = dict(zip(dfc["center"] + "|" + dfc["route"], dfc["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        key = str(row["center"]) + "|" + str(row["route"])
        _id = id_map.get(key)
        if _id and bool(row["완료"]): new_done.add(_id)
    st.session_state.done_center_ids = new_done
    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)

    # 세그먼트(센터 i → i+1) 구성: 거리는 (1) leg_km > (2) 좌표 하버사인 > (3) 0
    seg_rows = []
    for route, g in dfc.groupby("route"):
        g = g.sort_values("seq")
        recs = g.to_dict("records")
        for i in range(len(recs)-1):
            a, b = recs[i], recs[i+1]
            # 우선순위 1: leg_km
            if not pd.isna(a.get("leg_km")):
                dist = float(a["leg_km"])
            else:
                dist = haversine_km(a.get("lat"),a.get("lng"), b.get("lat"),b.get("lng"))
                if pd.isna(dist): dist = 0.0
            seg_rows.append({
                "route": route,
                "start_center": a["center"], "end_center": b["center"],
                "start_lat": a.get("lat"), "start_lng": a.get("lng"),
                "end_lat": b.get("lat"), "end_lng": b.get("lng"),
                "distance_km": dist,
                "done": bool(a["완료"] and b["완료"]),
            })
    seg_df = pd.DataFrame(seg_rows)

    # KPI(센터 기준)
    if not seg_df.empty:
        total_km_centers = float(seg_df["distance_km"].sum())
        done_km_centers  = float(seg_df.loc[seg_df["done"], "distance_km"].sum())
        left_km_centers  = max(total_km_centers - done_km_centers, 0.0)
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
        for flag, color in [(True,[28,200,138]), (False,[230,57,70])]:
            src = seg_df[seg_df["done"]==flag].copy()
            if src.empty: continue
            src["__path"] = src.apply(lambda r: [[r["start_lng"], r["start_lat"]],
                                                 [r["end_lng"], r["end_lat"]]], axis=1)
            src["__color"] = [color]*len(src)
            layers.append(pdk.Layer("PathLayer", src, get_path="__path", get_color="__color",
                                    width_scale=3, width_min_pixels=3, pickable=True))
    geo = dfc.dropna(subset=["lat","lng"]).copy()
    if not geo.empty:
        geo["__color"] = geo["완료"].map(lambda b: [28,200,138] if b else [230,57,70])
        layers.append(pdk.Layer("ScatterplotLayer",
                                geo.rename(columns={"lat":"latitude","lng":"longitude"}),
                                get_position='[longitude, latitude]', get_fill_color="__color",
                                get_radius=180, pickable=True))
        vlat, vlng = float(geo["latitude"].mean()), float(geo["longitude"].mean())
    else:
        vlat, vlng = 36.2, 127.5

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=7),
        tooltip={"text": "{route}\n{start_center} → {end_center}"}
    ), use_container_width=True)

    st.info("✅ 좌표가 비어도 낙동강 기본좌표로 자동 보정합니다. 필요시 centers.csv에 더 정확한 좌표/leg_km(구간거리)을 넣으면 그 값을 우선 사용합니다.")
