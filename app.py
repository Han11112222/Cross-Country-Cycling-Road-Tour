# app.py — v19 (구간+인증센터, 1클릭 반영, 회색배경+컬러오버레이 안정화)
from __future__ import annotations

import json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

BUILD_TAG = "2025-09-01-v19"

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")
st.caption(f"BUILD: {BUILD_TAG}")

# ─────────────────────────────────────────────────────────────
# 공식 총거리
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
# 분류/명칭 표준화
# ─────────────────────────────────────────────────────────────
TOP_ORDER = ["국토종주", "4대강 종주", "그랜드슬램", "제주환상"]
BIG_TO_ROUTES = {
    "국토종주": ["아라자전거길", "한강종주자전거길(서울구간)", "남한강자전거길", "새재자전거길", "낙동강자전거길"],
    "4대강 종주": ["한강종주자전거길(서울구간)", "금강자전거길", "영산강자전거길", "낙동강자전거길"],
    "그랜드슬램": ["북한강자전거길", "섬진강자전거길", "오천자전거길", "동해안자전거길(강원구간)", "동해안자전거길(경북구간)"],
    "제주환상": ["제주환상", "제주환상자전거길"],
}
def norm_name(s: str) -> str:
    s = str(s).strip()
    return "제주환상" if s == "제주환상자전거길" else s
ROUTE_TO_BIG = {norm_name(r): big for big, rs in BIG_TO_ROUTES.items() for r in rs}
ALL_DEFINED_ROUTES = sorted({norm_name(r) for v in BIG_TO_ROUTES.values() for r in v})

# ─────────────────────────────────────────────────────────────
# 폴백 경로([lng,lat])
# ─────────────────────────────────────────────────────────────
_raw_fb = {
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
    "제주환상": [[126.32, 33.50], [126.70, 33.52], [126.95, 33.45], [126.95, 33.25], [126.60, 33.23], [126.32, 33.35], [126.32, 33.50]],
}
FALLBACK_PATHS = {norm_name(k): v for k, v in _raw_fb.items()}

# ─────────────────────────────────────────────────────────────
# 유틸/지오코딩/뷰 계산(안전)
# ─────────────────────────────────────────────────────────────
def haversine_km(a, b, c, d):
    if any(pd.isna([a, b, c, d])): return np.nan
    R = 6371.0088
    p1, p2 = math.radians(a), math.radians(c)
    dphi, dlambda = math.radians(c - a), math.radians(d - b)
    x = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))

def parse_path(s):
    try:
        v = json.loads(s)
        if isinstance(v, list): return v
    except Exception:
        pass
    return None

@st.cache_data(ttl=60*60*24)
def geocode(addr: str):
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": addr, "format": "json", "limit": 1},
            headers={"User-Agent": "ccct/1.0"},
            timeout=10,
        )
        if r.ok and r.json():
            j = r.json()[0]
            return float(j["lat"]), float(j["lon"])
    except Exception:
        pass
    return None, None

def view_from_safe(paths, base_zoom: float, centers_df=None):
    """paths: [ [ [lng,lat], ... ], ... ] -> (lat,lng,zoom)"""
    pts = []
    for p in (paths or []):
        for xy in (p or []):
            try:
                lng, lat = float(xy[0]), float(xy[1])
                if not (np.isnan(lng) or np.isnan(lat)):
                    pts.append([lat, lng])
            except Exception:
                continue
    if centers_df is not None:
        try:
            pts += centers_df[["lat", "lng"]].dropna().astype(float).values.tolist()
        except Exception:
            pass
    if not pts:
        return 36.2, 127.5, base_zoom
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    vlat = float(np.nanmean(arr[:, 0])); vlng = float(np.nanmean(arr[:, 1]))
    span = max(float(np.nanmax(arr[:, 0]) - np.nanmin(arr[:, 0])),
               float(np.nanmax(arr[:, 1]) - np.nanmin(arr[:, 1])))
    zoom = 6.0 if span > 3.0 else base_zoom
    return vlat, vlng, zoom

def make_geojson_lines(items):
    """items: [{'route':str,'path':[[lng,lat],...],'color':[r,g,b], 'width':int}, ...]"""
    feats = []
    for it in (items or []):
        coords = it.get("path") or []
        if not isinstance(coords, list) or len(coords) < 2: continue
        try:
            _ = [[float(x), float(y)] for x, y in coords]
        except Exception:
            continue
        feats.append({
            "type": "Feature",
            "properties": {
                "route": it.get("route", ""),
                "color": (it.get("color") or [28, 200, 138]) + [255],
                "width": int(it.get("width") or 4),
                "start_center": it.get("start_center", ""),
                "end_center": it.get("end_center", ""),
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        })
    return {"type": "FeatureCollection", "features": feats}

# ─────────────────────────────────────────────────────────────
# CSV 로딩
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_routes(src):
    df = pd.read_csv(src)
    need = {"route", "section", "distance_km"}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"routes.csv 필요 컬럼: {sorted(miss)}")
    df["route"] = df["route"].astype(str).str.strip().map(norm_name)
    df["section"] = df["section"].astype(str).str.strip()
    df["distance_km"] = pd.to_numeric(df["distance_km"], errors="coerce")
    if "id" not in df.columns:
        df["id"] = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)
    df["big"] = df["route"].map(ROUTE_TO_BIG).fillna("기타")
    df["big"] = pd.Categorical(df["big"], categories=TOP_ORDER, ordered=True)
    if "path" in df.columns:
        m = df["path"].notna()
        df.loc[m, "path"] = df.loc[m, "path"].map(parse_path)
    return df

@st.cache_data
def load_centers(src, auto_geo: bool):
    if src is None: return None
    df = pd.read_csv(src)
    need = {"route", "center", "address", "lat", "lng", "id", "seq"}
    miss = need - set(df.columns)
    if miss: raise ValueError(f"centers.csv 필요 컬럼: {sorted(miss)}")
    df["route"] = df["route"].astype(str).str.strip().map(norm_name)
    for c in ["center", "address", "id"]:
        df[c] = df[c].astype(str).str.strip()
    for c in ["lat", "lng", "seq", "leg_km"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if auto_geo:
        needs = df[df["address"].notna() & (df["lat"].isna() | df["lng"].isna())]
        for i, row in needs.iterrows():
            lat, lng = geocode(row["address"])
            if lat is not None and lng is not None:
                df.at[i, "lat"], df.at[i, "lng"] = lat, lng
                time.sleep(1.0)  # OSM rate limit
    df["big"] = df["route"].map(ROUTE_TO_BIG).fillna("기타")
    df["big"] = pd.Categorical(df["big"], categories=TOP_ORDER, ordered=True)
    return df

# ─────────────────────────────────────────────────────────────
# 데이터/옵션
# ─────────────────────────────────────────────────────────────
st.sidebar.header("데이터")
use_repo = st.sidebar.radio("불러오기 방식", ["Repo 내 파일", "CSV 업로드"], index=0)
auto_geo = st.sidebar.toggle("주소 → 좌표 자동보정(지오코딩)", value=True)
show_debug = st.sidebar.checkbox("디버그 보기", value=False)
if st.sidebar.button("↻ 캐시 초기화", use_container_width=True):
    st.cache_data.clear(); st.rerun()

if use_repo == "Repo 내 파일":
    routes = load_routes(Path("data/routes.csv"))
    centers = load_centers(Path("data/centers.csv"), auto_geo) if Path("data/centers.csv").exists() else None
else:
    r_up = st.sidebar.file_uploader("routes.csv 업로드", type=["csv"], key="routes_up")
    c_up = st.sidebar.file_uploader("centers.csv 업로드(선택)", type=["csv"], key="centers_up")
    if r_up is None:
        st.info("routes.csv를 올리면 시작합니다."); st.stop()
    routes = load_routes(r_up)
    centers = load_centers(c_up, auto_geo) if c_up else None

# 완료 상태 저장
st.session_state.setdefault("done_section_ids", set())
st.session_state.setdefault("done_center_ids", set())

# 색상
ROUTE_COLORS = {
    "아라자전거길": [0, 173, 181],
    "한강종주자전거길(서울구간)": [0, 122, 255],
    "남한강자전거길": [88, 86, 214],
    "새재자전거길": [255, 159, 10],
    "낙동강자전거길": [255, 45, 85],
    "금강자전거길": [255, 204, 0],
    "영산강자전거길": [52, 199, 89],
    "북한강자전거길": [142, 142, 147],
    "섬진강자전거길": [175, 82, 222],
    "오천자전거길": [255, 55, 95],
    "동해안자전거길(강원구간)": [90, 200, 250],
    "동해안자전거길(경북구간)": [0, 199, 190],
    "제주환상": [255, 69, 0],
}

# ─────────────────────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────────────────────
tab = st.radio("", ["🚴 구간(거리) 추적", "📍 인증센터"], horizontal=True, label_visibility="collapsed")

def pick_by_big(all_routes: list[str], key_prefix: str, use_defined=True):
    big = st.sidebar.selectbox("대분류", TOP_ORDER, index=0, key=f"{key_prefix}_big")
    defined = [norm_name(r) for r in BIG_TO_ROUTES.get(big, [])] if use_defined else all_routes
    present = [r for r in defined if r in all_routes]
    absent  = [r for r in defined if r not in all_routes]
    options = present + [r for r in absent if r in ALL_DEFINED_ROUTES]
    fmt = lambda r: r if r in present else f"{r}  • 데이터없음(폴백)"
    picked = st.sidebar.multiselect("노선(복수 선택 가능)", options, default=present or options[:1],
                                    format_func=fmt, key=f"{key_prefix}_routes")
    return big, [norm_name(r) for r in picked]

# ─────────────────────────────────────────────────────────────
# 1) 구간(거리) 추적
# ─────────────────────────────────────────────────────────────
if tab == "🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")
    all_route_names = sorted(routes["route"].unique().tolist())
    big, picked = pick_by_big(all_route_names + ALL_DEFINED_ROUTES, "seg", use_defined=True)

    # routes.path 파싱 보정
    routes2 = routes.copy()
    if "path" in routes2.columns:
        m = routes2["path"].notna()
        routes2.loc[m, "path"] = routes2.loc[m, "path"].map(parse_path)

    # 노선별 경로(우선순위: routes.path > centers > fallback)
    def best_path_for(rname: str):
        sub = routes2[routes2["route"] == rname]
        if not sub.empty and sub["path"].notna().any():
            p = sub["path"].dropna().iloc[0]
            if isinstance(p, list) and len(p) >= 2:
                return p
        if centers is not None:
            g = centers[(centers["route"] == rname)].dropna(subset=["lat", "lng"]).sort_values("seq")
            if not g.empty:
                pts = g[["lng", "lat"]].to_numpy(float).tolist()
                if len(pts) >= 2:
                    return pts
        return FALLBACK_PATHS.get(rname)

    # 표: 체크박스 1클릭 즉시 반영 (id를 인덱스로 고정)
    base = routes[routes["route"].isin(picked)][["id", "route", "section", "distance_km"]].copy()
    base["완료"] = base["id"].isin(st.session_state.done_section_ids)
    view_df = base.set_index("id", drop=False)
    edited = st.data_editor(
        view_df[["route", "section", "distance_km", "완료"]],
        column_config={"완료": st.column_config.CheckboxColumn(label="완료", default=False)},
        use_container_width=True, hide_index=True, key="editor_routes",
    )
    # 1클릭 → new_done 집합 계산
    st.session_state.done_section_ids = set(edited.index[edited["완료"]].tolist())
    base["완료"] = base["id"].isin(st.session_state.done_section_ids)

    # 합계
    def displayed_total_km(routes_df: pd.DataFrame, fallback_names: list[str]) -> float:
        if not routes_df.empty:
            s = float(routes_df["distance_km"].fillna(0).sum())
            if s > 0: return s
        return float(sum(OFFICIAL_TOTALS.get(r, 0.0) for r in fallback_names))

    total_km = displayed_total_km(base, picked)
    done_km  = float(base.loc[base["완료"], "distance_km"].fillna(0).sum())
    left_km  = max(total_km - done_km, 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("대분류", big)

    # 지도: 전체 경로 회색 + 완료(해당 노선에 1개라도 체크)면 컬러 오버레이
    bg_rows, fg_rows, view_paths = [], [], []
    for r in picked:
        p = best_path_for(r)
        if not p or len(p) < 2:
            continue
        view_paths.append(p)
        # 회색 배경
        bg_rows.append({"route": r, "path": p, "color": [200, 200, 200], "width": 5})
        # 완료 여부
        any_done = base[(base["route"] == r)]["완료"].any()
        if any_done:
            fg_rows.append({"route": r, "path": p, "color": ROUTE_COLORS.get(r, [28, 200, 138]), "width": 6})

    gj_bg = make_geojson_lines(bg_rows)
    gj_fg = make_geojson_lines(fg_rows)

    layers = []
    if gj_bg["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_bg,
                                pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=4))
    if gj_fg["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_fg,
                                pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=6))

    # 인증센터 점(淡灰)
    centers_for_view = None
    if centers is not None:
        g = centers[centers["route"].isin(picked)].dropna(subset=["lat", "lng"]).copy()
        if not g.empty:
            centers_for_view = g.copy()
            g["__color"] = [[220, 220, 220]] * len(g)
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                g.rename(columns={"lat": "latitude", "lng": "longitude"}),
                get_position='[longitude, latitude]',
                get_fill_color="__color",
                get_radius=140,
                pickable=True,
            ))

    vlat, vlng, vzoom = view_from_safe(view_paths, 7.0 if len(picked) == 1 else 5.8, centers_for_view)
    if show_debug:
        st.write({"bg": len(gj_bg["features"]), "fg": len(gj_fg["features"]), "paths_for_view": len(view_paths)})

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text": "{properties.route}"},
    ), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 2) 인증센터
# ─────────────────────────────────────────────────────────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다."); st.stop()

    st.sidebar.header("인증센터 필터")
    _, picked = pick_by_big(sorted(set(routes["route"]) | set(centers["route"]) | set(ALL_DEFINED_ROUTES)), "cent", use_defined=True)

    dfc = centers[centers["route"].isin(picked)].copy().sort_values(["route", "seq", "center"]).reset_index(drop=True)
    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)

    # 체크 1클릭 → id를 인덱스로
    cols = ["route", "seq", "center", "address", "완료"]
    view_centers = dfc.set_index("id", drop=False)
    with st.expander("인증센터 체크(간단 편집)", expanded=True):
        edited = st.data_editor(
            view_centers[cols],
            column_config={"완료": st.column_config.CheckboxColumn(label="완료", default=False)},
            use_container_width=True, hide_index=True, key="editor_centers",
        )
    st.session_state.done_center_ids = set(edited.index[edited["완료"]].tolist())
    dfc["완료"] = dfc["id"].isin(st.session_state.done_center_ids)

    # 센터 구간(연속 센터 사이)을 만들고, 양 끝 센터 모두 완료이면 done
    seg = []
    for r, g in dfc.groupby("route"):
        g = g.sort_values("seq")
        rec = g.to_dict("records")
        for i in range(len(rec) - 1):
            a, b = rec[i], rec[i + 1]
            if any(pd.isna([a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng")])):  # 좌표 없는 경우 skip
                continue
            dist = a.get("leg_km")
            if pd.isna(dist):
                dist = haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])
            seg.append({
                "route": r,
                "start_center": a["center"], "end_center": b["center"],
                "path": [[float(a["lng"]), float(a["lat"])], [float(b["lng"]), float(b["lat"])]],
                "distance_km": float(dist or 0.0),
                "done": bool(a["완료"] and b["완료"]),
            })
    seg_df = pd.DataFrame(seg)

    total = float(seg_df["distance_km"].sum()) if not seg_df.empty else 0.0
    done  = float(seg_df.loc[seg_df["done"], "distance_km"].sum()) if not seg_df.empty else 0.0
    left  = max(total - done, 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 인증센터 수", f"{dfc.shape[0]:,}")
    c2.metric("완료한 인증센터", f"{int(dfc['완료'].sum()):,}")
    c3.metric("센터 기준 누적거리", f"{done:,.1f} km")
    c4.metric("센터 기준 남은 거리", f"{left:,.1f} km")

    done_items = [{"route": r["route"], "path": r["path"], "color": [28, 200, 138], "width": 4}
                  for _, r in seg_df[seg_df["done"]].iterrows()]
    todo_items = [{"route": r["route"], "path": r["path"], "color": [230, 57, 70], "width": 4}
                  for _, r in seg_df[~seg_df["done"]].iterrows()]
    gj_done = make_geojson_lines(done_items)
    gj_todo = make_geojson_lines(todo_items)

    layers = []
    if gj_todo["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_todo,
                                pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=4))
    if gj_done["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_done,
                                pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=4))

    # 센터 점(완료=초록, 미완=빨강)
    geo = dfc.dropna(subset=["lat", "lng"]).copy()
    if not geo.empty:
        geo["__color"] = geo["완료"].map(lambda b: [28, 200, 138] if b else [230, 57, 70])
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            geo.rename(columns={"lat": "latitude", "lng": "longitude"}),
            get_position='[longitude, latitude]',
            get_fill_color="__color",
            get_radius=160,
            pickable=True,
        ))

    vlat, vlng, vzoom = view_from_safe([], 7.0, geo)
    if show_debug:
        st.write({"center_segments_total": len(seg_df),
                  "gj_done_features": len(gj_done["features"]),
                  "gj_todo_features": len(gj_todo["features"])})
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text": "{properties.route}\n{properties.start_center} → {properties.end_center}"},
    ), use_container_width=True)
