# app.py — 지도 뷰포인트 고정(모든 경로 평균), 인증센터 latitude KeyError 제거
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# ─────────────────────────────────────────────────────────────
# 0) 공식 총거리(자전거행복나눔 기준)
# ─────────────────────────────────────────────────────────────
OFFICIAL_TOTALS = {
    "아라자전거길": 21,
    "한강종주자전거길(서울구간)": 40,
    "남한강자전거길": 132,
    "새재자전거길": 100,
    "낙동강자전거길": 389,
    "금강자전거길": 146,
    "영산강자전거길": 133,
    "북한강자전거길": 70,
    "섬진강자전거길": 148,
    "오천자전거길": 105,
    "동해안자전거길(강원구간)": 242,
    "동해안자전거길(경북구간)": 76,
    "제주환상": 234,
    "제주환상자전거길": 234,
}

# ─────────────────────────────────────────────────────────────
# 1) 대·중 분류(고정) + 명칭 표준화
# ─────────────────────────────────────────────────────────────
TOP_ORDER = ["국토종주", "4대강 종주", "그랜드슬램", "제주환상"]

BIG_TO_ROUTES = {
    "국토종주": [
        "아라자전거길",
        "한강종주자전거길(서울구간)",
        "남한강자전거길",
        "새재자전거길",
        "낙동강자전거길",
    ],
    "4대강 종주": [
        "한강종주자전거길(서울구간)",
        "금강자전거길",
        "영산강자전거길",
        "낙동강자전거길",
    ],
    "그랜드슬램": [
        "북한강자전거길",
        "섬진강자전거길",
        "오천자전거길",
        "동해안자전거길(강원구간)",
        "동해안자전거길(경북구간)",
    ],
    "제주환상": ["제주환상", "제주환상자전거길"],
}

def normalize_route_name(name: str) -> str:
    n = str(name).strip()
    if n == "제주환상자전거길":
        return "제주환상"
    return n

ROUTE_TO_BIG = {}
for big, routes in BIG_TO_ROUTES.items():
    for r in routes:
        ROUTE_TO_BIG[normalize_route_name(r)] = big

ALL_DEFINED_ROUTES = sorted({normalize_route_name(r) for v in BIG_TO_ROUTES.values() for r in v})

# ─────────────────────────────────────────────────────────────
# 2) 폴백 경로 — [lng, lat]
# ─────────────────────────────────────────────────────────────
FALLBACK_PATHS = {
    "아라자전거길": [[126.58, 37.60], [126.68, 37.60], [126.82, 37.57]],
    "한강종주자전거길(서울구간)": [[126.82, 37.57], [127.02, 37.55], [127.08, 37.54]],
    "남한강자전거길": [[127.31, 37.55], [127.63, 37.29], [127.90, 36.98]],
    "새재자전거길": [[127.90, 36.98], [128.07, 36.69], [128.16, 36.41]],
    "낙동강자전거길": [[128.72, 36.56], [128.60, 35.87], [128.50, 35.40], [129.03, 35.10]],
    "금강자전거길": [[127.48, 36.44], [127.28, 36.50], [127.12, 36.45], [126.71, 36.00]],
    "영산강자전거길": [[126.99, 35.32], [126.72, 35.02], [126.39, 34.79]],
    "북한강자전거길": [[127.31, 37.55], [127.63, 37.74], [127.73, 37.88]],
    "섬진강자전거길": [[127.38, 35.41], [127.47, 35.22], [127.75, 35.10], [127.69, 34.94]],
    "오천자전거길": [[126.60, 36.33], [126.85, 36.40], [127.12, 36.45]],
    "동해안자전거길(강원구간)": [[128.45, 38.38], [128.60, 38.20], [129.00, 37.75], [129.20, 37.44]],
    "동해안자전거길(경북구간)": [[129.20, 37.44], [129.36, 36.03], [129.31, 35.84], [129.35, 35.55]],
    "제주환상": [[126.32, 33.50], [126.70, 33.52], [126.95, 33.45], [126.95, 33.25],
             [126.60, 33.23], [126.32, 33.35], [126.32, 33.50]],
}

# ─────────────────────────────────────────────────────────────
# 3) 유틸 + 지오코딩
# ─────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])): return np.nan
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def parse_path(s):
    try:
        v = json.loads(s)
        if isinstance(v, list): return v
    except Exception:
        pass
    return None

@st.cache_data(ttl=60*60*24)
def geocode_osm(address: str) -> tuple[float|None, float|None]:
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1},
            headers={"User-Agent": "cross-country-cycling-tracker/1.0"},
            timeout=10,
        )
        if resp.ok:
            js = resp.json()
            if js:
                return float(js[0]["lat"]), float(js[0]["lon"])
    except Exception:
        return None, None
    return None, None

# 모든 경로/센터에서 뷰포인트 계산(여러 노선 선택해도 전체가 보이도록)
def view_from_paths_and_centers(paths: list[list[list[float]]], centers_df: pd.DataFrame | None, default_zoom: float) -> tuple[float, float, float]:
    pts = []
    for p in paths:
        if isinstance(p, list) and p:
            for xy in p:
                if isinstance(xy, (list, tuple)) and len(xy) == 2 and not any(pd.isna(xy)):
                    # [lng, lat] → [lat, lng]
                    pts.append([float(xy[1]), float(xy[0])])
    if centers_df is not None and not centers_df.empty:
        pts += centers_df[["lat","lng"]].dropna().astype(float).values.tolist()
    if pts:
        arr = np.array(pts, dtype=float)
        vlat, vlng = float(arr[:,0].mean()), float(arr[:,1].mean())
        zoom = default_zoom
        # 좌표 범위에 따라 줌을 조금 보정
        lat_span = float(arr[:,0].max() - arr[:,0].min())
        lng_span = float(arr[:,1].max() - arr[:,1].min())
        if max(lat_span, lng_span) > 3:  # 전국적 범위면 더 넓게
            zoom = min(zoom, 6.0)
        return vlat, vlng, zoom
    return 36.2, 127.5, default_zoom

# ─────────────────────────────────────────────────────────────
# 4) CSV 로딩
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_routes(src) -> pd.DataFrame:
    df = pd.read_csv(src)
    need = {"route", "section", "distance_km"}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"routes.csv에 필요 컬럼: {sorted(miss)}")
    df["route"] = df["route"].astype(str).map(normalize_route_name)
    for c in ["section", "start", "end"]:
        if c in df.columns: df[c] = df[c].astype(str).str.strip()
    for c in ["distance_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if "id" not in df.columns:
        df["id"] = (df["route"].astype(str)+"@"+df["section"].astype(str)).str.replace(r"\s+","",regex=True)
    df["big"] = df["route"].map(ROUTE_TO_BIG).fillna("기타")
    df["big"] = pd.Categorical(df["big"], categories=TOP_ORDER, ordered=True)
    return df

@st.cache_data
def load_centers(src, auto_geocode: bool) -> pd.DataFrame | None:
    if src is None: return None
    df = pd.read_csv(src)
    need = {"route", "center", "address", "lat", "lng", "id", "seq"}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"centers.csv에 필요 컬럼: {sorted(miss)}")
    df["route"] = df["route"].astype(str).map(normalize_route_name)
    for c in ["center", "address", "id"]:
        df[c] = df[c].astype(str).str.strip()
    for c in ["lat", "lng", "seq", "leg_km"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    if auto_geocode and ("address" in df.columns):
        needs = df[df["address"].notna() & (df["lat"].isna() | df["lng"].isna())]
        for idx, row in needs.iterrows():
            lat, lng = geocode_osm(row["address"])
            if lat is not None and lng is not None:
                df.at[idx, "lat"] = lat
                df.at[idx, "lng"] = lng
                time.sleep(1.0)
    df["big"] = df["route"].map(ROUTE_TO_BIG).fillna("기타")
    df["big"] = pd.Categorical(df["big"], categories=TOP_ORDER, ordered=True)
    return df

# ─────────────────────────────────────────────────────────────
# 5) 데이터 소스 + 캐시 초기화
# ─────────────────────────────────────────────────────────────
st.sidebar.header("데이터")
use_repo = st.sidebar.radio("불러오기 방식", ["Repo 내 파일", "CSV 업로드"], index=0)
auto_geocode_on = st.sidebar.toggle("주소 → 좌표 자동보정(지오코딩)", value=True)
if st.sidebar.button("↻ 캐시 초기화", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if use_repo == "Repo 내 파일":
    routes_csv = Path("data/routes.csv")
    centers_csv = Path("data/centers.csv")
    if not routes_csv.exists():
        st.error("Repo에 data/routes.csv 가 없습니다.")
        st.stop()
    routes = load_routes(routes_csv)
    centers = load_centers(centers_csv, auto_geocode_on) if centers_csv.exists() else None
else:
    up_r = st.sidebar.file_uploader("routes.csv 업로드", type=["csv"], key="routes_up")
    up_c = st.sidebar.file_uploader("centers.csv 업로드(선택)", type=["csv"], key="centers_up")
    if up_r is None:
        st.info("routes.csv를 올리면 시작합니다.")
        st.stop()
    routes = load_routes(up_r)
    centers = load_centers(up_c, auto_geocode_on) if up_c else None

st.session_state.setdefault("done_section_ids", set())
st.session_state.setdefault("done_center_ids", set())

# ─────────────────────────────────────────────────────────────
# 6) 탭
# ─────────────────────────────────────────────────────────────
tab = st.radio("", ["🚴 구간(거리) 추적", "📍 인증센터"], horizontal=True, label_visibility="collapsed")

def big_and_routes_selector(source_routes: list[str], key_prefix: str, use_defined: bool=False):
    big = st.sidebar.selectbox("대분류", TOP_ORDER, index=0, key=f"{key_prefix}_big")
    defined = [normalize_route_name(r) for r in BIG_TO_ROUTES.get(big, [])] if use_defined else source_routes
    present = [r for r in defined if r in source_routes]
    absent  = [r for r in defined if r not in source_routes]
    options = present + [r for r in absent if r in ALL_DEFINED_ROUTES]
    def _fmt(r): return r if r in present else f"{r}  • 데이터없음(폴백)"
    picked = st.sidebar.multiselect("노선(복수 선택 가능)", options=options,
                                    default=present[:1] if present else options[:1],
                                    format_func=_fmt, key=f"{key_prefix}_routes")
    return big, [normalize_route_name(r) for r in picked]

# ─────────────────────────────────────────────────────────────
# 7) 구간(거리) 추적
# ─────────────────────────────────────────────────────────────
if tab == "🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")

    all_route_names = sorted(routes["route"].unique().tolist())
    big, picked_routes = big_and_routes_selector(all_route_names + ALL_DEFINED_ROUTES, key_prefix="seg", use_defined=True)

    df = routes[routes["route"].isin(picked_routes)].copy()
    df["__path"] = None
    if "path" in df.columns:
        m = df["path"].notna()
        df.loc[m, "__path"] = df.loc[m, "path"].map(parse_path)

    def centers_polyline_and_km(route_name: str):
        if centers is None: return None, np.nan
        g = centers[(centers["route"] == route_name)].dropna(subset=["lat","lng"]).sort_values("seq")
        if g.empty: return None, np.nan
        pts = g[["lng","lat"]].to_numpy(dtype=float)
        path = pts.tolist()
        km = float(g["leg_km"].fillna(0).sum()) if ("leg_km" in g.columns and g["leg_km"].notna().any()) else \
             sum(haversine_km(pts[i][1], pts[i][0], pts[i+1][1], pts[i+1][0]) for i in range(len(pts)-1))
        return path, km

    agg_rows = []
    for rname in picked_routes:
        sub = df[df["route"] == rname]
        km_routes = float(sub["distance_km"].fillna(0).sum()) if not sub.empty else 0.0
        path = None
        if not sub.empty and sub["__path"].notna().any():
            path = sub["__path"].dropna().iloc[0]
        if path is None:
            p2, k2 = centers_polyline_and_km(rname)
            path = p2
            if km_routes == 0 and not np.isnan(k2): km_routes = k2
        if path is None:
            path = FALLBACK_PATHS.get(rname)
        display_km = km_routes if km_routes > 0 else float(OFFICIAL_TOTALS.get(rname, 0.0))
        agg_rows.append({"route": rname, "display_km": display_km, "path": path})
    agg = pd.DataFrame(agg_rows)

    with st.expander("선택 노선 총거리 요약", expanded=True):
        st.dataframe(agg[["route","display_km"]].rename(columns={"display_km":"표시거리(km)"}),
                     use_container_width=True, hide_index=True)

    base = routes[routes["route"].isin(picked_routes)][["route","section","distance_km","id"]].copy()
    base["완료"] = base["id"].isin(st.session_state.done_section_ids)
    edited = st.data_editor(base.drop(columns=["id"]), use_container_width=True, hide_index=True, key="editor_routes")
    id_map = dict(zip(base["route"].astype(str)+"@"+base["section"].astype(str), base["id"]))
    new_done = set()
    for _, row in edited.iterrows():
        key = f"{row['route']}@{row['section']}"
        if id_map.get(key) and bool(row["완료"]): new_done.add(id_map[key])
    st.session_state.done_section_ids = new_done
    base["완료"] = base["id"].isin(st.session_state.done_section_ids)

    total_km = float(base["distance_km"].fillna(0).sum()) if not base.empty else 0.0
    if total_km == 0:
        total_km = float(agg["display_km"].fillna(0).sum())
    done_km = float(base.loc[base["완료"],"distance_km"].fillna(0).sum())
    if done_km == 0 and total_km > 0 and not base.empty:
        done_km = total_km * float(base["완료"].mean())
    left_km = max(total_km - done_km, 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("대분류", big)

    # ── 지도(여러 노선 동시 표시 + 전체 평균 뷰포인트)
    layers = []
    draw = agg.dropna(subset=["path"]).copy()
    if not draw.empty:
        draw["__color"] = [[28,200,138]] * len(draw)
        layers.append(pdk.Layer("PathLayer", draw, get_path="path", get_color="__color",
                                width_scale=3, width_min_pixels=3, pickable=True))
    centers_for_view = None
    if centers is not None:
        centers_for_view = centers[centers["route"].isin(picked_routes)].dropna(subset=["lat","lng"]).copy()
        if not centers_for_view.empty:
            g = centers_for_view.copy()
            g["__color"] = [[200,200,200]] * len(g)
            layers.append(pdk.Layer("ScatterplotLayer",
                                    g.rename(columns={"lat":"latitude","lng":"longitude"}),
                                    get_position='[longitude, latitude]',
                                    get_fill_color="__color", get_radius=120, pickable=True))
    vlat, vlng, vzoom = view_from_paths_and_centers(draw["path"].tolist(), centers_for_view, default_zoom=7.0 if len(picked_routes)==1 else 6.0)

    st.pydeck_chart(
        pdk.Deck(layers=layers,
                 initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
                 tooltip={"text": "{route}"}),
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────
# 8) 인증센터(소분류) — latitude KeyError 제거
# ─────────────────────────────────────────────────────────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다.")
        st.stop()

    st.sidebar.header("인증센터 필터")
    source_routes = sorted(set(routes["route"].tolist()) | set(centers["route"].tolist()) | set(ALL_DEFINED_ROUTES))
    _, picked_routes = big_and_routes_selector(source_routes, key_prefix="cent", use_defined=True)

    dfc = centers[centers["route"].isin(picked_routes)].copy()
    dfc = dfc.sort_values(["route","seq","center"]).reset_index(drop=True)
    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)

    with st.expander("인증센터 체크(간단 편집)", expanded=True):
        cols = ["route","seq","center","address","완료"]
        edited = st.data_editor(dfc[cols], use_container_width=True, hide_index=True, key="editor_centers")

    new_done = set()
    for i, row in edited.iterrows():
        cid = dfc.iloc[i]["id"]
        if bool(row["완료"]): new_done.add(cid)
    st.session_state.done_center_ids = new_done
    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)

    # 세그먼트
    seg_rows = []
    for rname, g in dfc.groupby("route"):
        g = g.sort_values("seq")
        recs = g.to_dict("records")
        for i in range(len(recs)-1):
            a, b = recs[i], recs[i+1]
            dist = float(a.get("leg_km")) if not pd.isna(a.get("leg_km")) else \
                   (haversine_km(a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng")) or 0.0)
            seg_rows.append({
                "route": rname,
                "start_center": a["center"], "end_center": b["center"],
                "start_lat": a.get("lat"), "start_lng": a.get("lng"),
                "end_lat": b.get("lat"), "end_lng": b.get("lng"),
                "distance_km": float(dist) if not pd.isna(dist) else 0.0,
                "done": bool(a["완료"] and b["완료"]),
            })
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
    c2.metric("완료한 인증센터", f"{int(dfc['완료'].sum()):,}")
    c3.metric("센터 기준 누적거리", f"{done_km_centers:,.1f} km")
    c4.metric("센터 기준 남은 거리", f"{left_km_centers:,.1f} km")

    # 지도
    layers = []
    if not seg_df.empty and seg_df[["start_lat","start_lng","end_lat","end_lng"]].notna().any().any():
        for flag, color in [(True,[28,200,138]), (False,[230,57,70])]:
            src = seg_df[seg_df["done"]==flag].dropna(subset=["start_lat","start_lng","end_lat","end_lng"]).copy()
            if src.empty: continue
            src["__path"] = src.apply(lambda r: [[r["start_lng"], r["start_lat"]],[r["end_lng"], r["end_lat"]]], axis=1)
            src["__color"] = [color]*len(src)
            layers.append(pdk.Layer("PathLayer", src, get_path="__path", get_color="__color",
                                    width_scale=3, width_min_pixels=3, pickable=True))

    geo = dfc.dropna(subset=["lat","lng"]).copy()
    if not geo.empty:
        geo["__color"] = geo["완료"].map(lambda b: [28,200,138] if b else [230,57,70])
        layers.append(pdk.Layer("ScatterplotLayer",
                                geo.rename(columns={"lat":"latitude","lng":"longitude"}),
                                get_position='[longitude, latitude]',
                                get_fill_color="__color", get_radius=160, pickable=True))

    # ★ KeyError 방지: 평균은 lat/lng로만 계산
    if not geo.empty:
        vlat, vlng, vzoom = view_from_paths_and_centers([], geo, default_zoom=7.0)
    else:
        picked = picked_routes[0] if picked_routes else None
        fb = FALLBACK_PATHS.get(picked) if picked else None
        vlat, vlng, vzoom = view_from_paths_and_centers([fb] if fb else [], None, default_zoom=7.0)

    st.pydeck_chart(
        pdk.Deck(layers=layers,
                 initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
                 tooltip={"text": "{route}\n{start_center} → {end_center}"}),
        use_container_width=True,
    )
