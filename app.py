# app.py — 국토종주 우선 정렬 + 거리 보정 + 센터경로 자동그리기
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 0) 공식 총거리(자전거행복나눔 등 공용값)
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# 1) 상위 카테고리(네가 요청한 3단 구조로 재분류)
#    - CSV의 category를 무시하고 아래 규칙으로 덮어쓴다
# ─────────────────────────────────────────────────────────────────────────────
GROUP_MAP = {
    # 국토종주코스
    "아라자전거길": "국토종주코스",
    "한강종주자전거길(서울구간)": "국토종주코스",
    "남한강자전거길": "국토종주코스",
    "새재자전거길": "국토종주코스",
    "낙동강자전거길": "국토종주코스",

    # 제주
    "제주환상": "제주환상자전거길",

    # 그랜드슬램코스(그 외 대표 코스들)
    "금강자전거길": "그랜드슬램코스",
    "영산강자전거길": "그랜드슬램코스",
    "동해안자전거길(강원구간)": "그랜드슬램코스",
    "동해안자전거길(경북구간)": "그랜드슬램코스",
    "섬진강자전거길": "그랜드슬램코스",
    "오천자전거길": "그랜드슬램코스",
    "북한강자전거길": "그랜드슬램코스",
}
TOP_ORDER = ["국토종주코스", "제주환상자전거길", "그랜드슬램코스", "기타코스"]  # 표시 우선순위

# ─────────────────────────────────────────────────────────────────────────────
# 2) 낙동강 인증센터 좌표(대략값) — 경로 그리기 fallback 용
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# 3) CSV 로더
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_routes(src: str | Path | bytes) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)

    need = {"route", "section", "distance_km"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"routes.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    # 정리
    for c in ["category", "route", "section", "start", "end"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["distance_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # id
    if "id" not in df.columns:
        df["id"] = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)

    # 상위 그룹 덮어쓰기
    df["category"] = df["route"].map(GROUP_MAP).fillna("기타코스")
    # 카테고리 정렬
    df["category"] = pd.Categorical(df["category"], categories=TOP_ORDER + ["기타코스"], ordered=True)

    return df


@st.cache_data
def load_centers(src: str | Path | bytes) -> pd.DataFrame:
    if isinstance(src, (str, Path)):
        df = pd.read_csv(src)
    else:
        df = pd.read_csv(src)

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

    # id 자동
    if "id" not in df.columns or df["id"].isna().any():
        df["id"] = np.where(
            df.get("id").isna() if "id" in df.columns else True,
            (df["route"] + "@" + df["center"]).str.replace(r"\s+", "", regex=True),
            df.get("id", "")
        )

    # 낙동강 좌표 보정
    for i, r in df.iterrows():
        _id = r.get("id")
        if (pd.isna(r.get("lat")) or pd.isna(r.get("lng"))) and _id in DEFAULT_CENTER_COORDS:
            df.loc[i, "lat"] = DEFAULT_CENTER_COORDS[_id][0]
            df.loc[i, "lng"] = DEFAULT_CENTER_COORDS[_id][1]

    # 상위 그룹 일치
    df["category"] = df["route"].map(GROUP_MAP).fillna("기타코스")
    df["category"] = pd.Categorical(df["category"], categories=TOP_ORDER + ["기타코스"], ordered=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4) 데이터 소스 선택
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# 5) 탭
# ─────────────────────────────────────────────────────────────────────────────
tab = st.radio("", ["🚴 구간(거리) 추적", "📍 인증센터"], horizontal=True, label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
# 6) 구간(거리) 추적
# ─────────────────────────────────────────────────────────────────────────────
if tab == "🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")

    # 국토종주 → 제주 → 그랜드슬램 순서로 보이게
    cat_list = ["전체구간"] + [c for c in TOP_ORDER if c in routes["category"].unique()]
    cat = st.sidebar.selectbox("대분류", options=cat_list, index=0)

    df = routes.copy()
    if cat != "전체구간":
        df = df[df["category"] == cat]
    # 보기 편하게 정렬
    df = df.sort_values(["category", "route", "section"]).reset_index(drop=True)

    # 노선 선택
    route_names = sorted(df["route"].dropna().unique().tolist())
    route_pick = st.sidebar.multiselect("노선(복수 선택 가능)", route_names, default=route_names)
    if not route_pick:
        st.stop()
    df = df[df["route"].isin(route_pick)].copy()

    # ---- (A) 거리 보정: 해당 노선의 distance_km 합이 0/NaN이면 OFFICIAL_TOTALS 로 표시용 컬럼 채움
    df["표시거리_km"] = df["distance_km"]
    def patch_route_total(g: pd.DataFrame) -> pd.DataFrame:
        rname = g["route"].iloc[0]
        s = pd.to_numeric(g["distance_km"], errors="coerce").fillna(0).sum()
        if (s == 0) and (rname in OFFICIAL_TOTALS):
            # 첫 행에만 총거리 넣고 나머진 0
            g = g.copy()
            g.loc[:, "표시거리_km"] = 0.0
            g.iloc[0, g.columns.get_loc("표시거리_km")] = float(OFFICIAL_TOTALS[rname])
        return g
    df = df.groupby("route", group_keys=False).apply(patch_route_total)

    # ---- (B) 완료 체크/반영
    if "done_ids" not in st.session_state:
        st.session_state.done_ids = set()
    df["완료"] = df["id"].isin(st.session_state.done_ids)

    edited = st.data_editor(
        df[["category", "route", "section", "표시거리_km", "완료"]],
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

    # ---- (C) KPI(표시거리 기준)
    total_km = float(pd.to_numeric(df["표시거리_km"], errors="coerce").fillna(0).sum())
    done_km  = float(pd.to_numeric(df[df["id"].isin(st.session_state.done_ids)]["표시거리_km"], errors="coerce").fillna(0).sum())
    left_km  = max(total_km - done_km, 0.0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 구간 총거리", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    if len(route_pick) == 1:
        c4.metric("공식 노선 총거리", f"{OFFICIAL_TOTALS.get(route_pick[0], total_km):,.1f} km")
    else:
        c4.metric("공식 노선 총거리", "다중 선택")

    # ---- (D) 지도
    # 1) CSV path 사용
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

    # 2) path가 없고, centers가 있을 때: 선택된 노선의 인증센터 좌표로 경로 생성(fallback)
    center_paths = []
    if centers is not None:
        cpick = centers[centers["route"].isin(route_pick)].dropna(subset=["lat", "lng"])
        for rname, g in cpick.groupby("route"):
            g = g.sort_values("seq")
            if len(g) >= 2:
                pts = g.apply(lambda r: [float(r["lng"]), float(r["lat"])], axis=1).tolist()
                center_paths.append({"id": f"{rname}-centers", "__path": pts})
    center_paths = pd.DataFrame(center_paths)

    # 3) 시작/끝 점 (없으면 생략)
    pts = []
    for _, r in df.iterrows():
        for (lng, lat, label) in [
            (r.get("start_lng"), r.get("start_lat"), "start"),
            (r.get("end_lng"),   r.get("end_lat"),   "end"),
        ]:
            if pd.notna(lng) and pd.notna(lat):
                pts.append({
                    "lng": float(lng), "lat": float(lat),
                    "name": f"{r['route']} / {r['section']} ({label})",
                    "done": bool(r["id"] in st.session_state.done_ids)
                })
    pts_df = pd.DataFrame(pts)

    # 중심
    if len(pts_df) > 0:
        center_lng, center_lat = float(pts_df["lng"].mean()), float(pts_df["lat"].mean())
    elif not center_paths.empty:
        coords = np.array(sum(center_paths["__path"].tolist(), []))  # flatten
        center_lng, center_lat = float(coords[:,0].mean()), float(coords[:,1].mean())
    else:
        center_lng, center_lat = 127.5, 36.2

    layers = []
    if not paths.empty:
        paths["__color"] = [ [230,57,70] ] * len(paths)   # 기본 빨강
        layers.append(pdk.Layer("PathLayer", paths, get_path="__path", get_color="__color",
                                width_scale=3, width_min_pixels=3, pickable=True))
    if not center_paths.empty:
        center_paths["__color"] = [ [200,200,200] ] * len(center_paths)  # 연한 회색 fallback
        layers.append(pdk.Layer("PathLayer", center_paths, get_path="__path", get_color="__color",
                                width_scale=2, width_min_pixels=2, pickable=True))
    if not pts_df.empty:
        pts_df["__color"] = pts_df["done"].map(lambda b: [28,200,138] if b else [230,57,70])
        layers.append(pdk.Layer("ScatterplotLayer", pts_df, get_position='[lng, lat]',
                                get_fill_color='__color', get_radius=150, pickable=True))

    st.pydeck_chart(
        pdk.Deck(layers=layers,
                 initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7),
                 tooltip={"text": "{name}"}),
        use_container_width=True,
    )
    st.caption("💡 거리칸이 비어있거나 0인 노선은 표에서 '공식 총거리'로 자동 보정됩니다. "
               "CSV에 path/좌표가 없으면 인증센터 좌표로 대략적인 경로를 그립니다(회색).")

# ─────────────────────────────────────────────────────────────────────────────
# 7) 인증센터(이전과 동일 — 누적거리 계산/경로 표출)
# ─────────────────────────────────────────────────────────────────────────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다.")
        st.stop()

    st.sidebar.header("인증센터 필터")
    cat_list = ["전체"] + [c for c in TOP_ORDER if c in centers["category"].unique()]
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

    dfc = dfc.sort_values(["route", "seq", "center"]).reset_index(drop=True)
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
        _id = id_map.get(str(row["center"]) + "|" + str(row["route"]))
        if _id and bool(row["완료"]):
            new_done.add(_id)
    st.session_state.done_center_ids = new_done
    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)

    # 세그먼트 거리(leg_km > 좌표거리 > 0)
    seg_rows = []
    for rname, g in dfc.groupby("route"):
        g = g.sort_values("seq")
        recs = g.to_dict("records")
        for i in range(len(recs)-1):
            a, b = recs[i], recs[i+1]
            if not pd.isna(a.get("leg_km")): dist = float(a["leg_km"])
            else:
                dist = haversine_km(a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng"))
                if pd.isna(dist): dist = 0.0
            seg_rows.append({
                "route": rname,
                "start_center": a["center"], "end_center": b["center"],
                "start_lat": a.get("lat"), "start_lng": a.get("lng"),
                "end_lat": b.get("lat"), "end_lng": b.get("lng"),
                "distance_km": dist,
                "done": bool(a["완료"] and b["완료"]),
            })
    seg_df = pd.DataFrame(seg_rows)

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
        layer_pts = pdk.Layer("ScatterplotLayer",
                              geo.rename(columns={"lat":"latitude","lng":"longitude"}),
                              get_position='[longitude, latitude]',
                              get_fill_color="__color",
                              get_radius=180,
                              pickable=True)
        layers.append(layer_pts)
        vlat, vlng = float(geo["latitude"].mean()), float(geo["longitude"].mean())
    else:
        vlat, vlng = 36.2, 127.5

    st.pydeck_chart(
        pdk.Deck(layers=layers,
                 initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=7),
                 tooltip={"text": "{route}\n{start_center} → {end_center}"}),
        use_container_width=True
    )
