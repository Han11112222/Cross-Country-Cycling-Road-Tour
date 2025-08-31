# app.py — v17
# 선택 없음=빈 지도, 선택(회색)=베이스라인, 완료(색)=오버레이, 즉시 반영(1클릭)
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

BUILD_TAG = "2025-09-01-v17"
st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")
st.caption(f"BUILD: {BUILD_TAG}")

# ─────────────────────────────────────────────────────────────
# 공식 총거리 (표시용 폴백)
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
    "국토종주": ["아라자전거길","한강종주자전거길(서울구간)","남한강자전거길","새재자전거길","낙동강자전거길"],
    "4대강 종주": ["한강종주자전거길(서울구간)","금강자전거길","영산강자전거길","낙동강자전거길"],
    "그랜드슬램": ["북한강자전거길","섬진강자전거길","오천자전거길","동해안자전거길(강원구간)","동해안자전거길(경북구간)"],
    "제주환상": ["제주환상","제주환상자전거길"],
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
    "제주환상": [[126.32, 33.50], [126.70, 33.52], [126.95, 33.45], [126.95, 33.25],[126.60, 33.23], [126.32, 33.35], [126.32, 33.50]],
}
FALLBACK_PATHS = {norm_name(k): v for k, v in _raw_fb.items()}

# ─────────────────────────────────────────────────────────────
# 유틸/지오코딩/뷰
# ─────────────────────────────────────────────────────────────
def parse_path(s):
    try:
        v=json.loads(s)
        if isinstance(v,list): return v
    except Exception:
        pass
    return None

def view_from_safe(paths, centers_df, base_zoom: float):
    pts = []
    for p in (paths or []):
        for xy in (p or []):
            try:
                lng, lat = float(xy[0]), float(xy[1])
                if not (np.isnan(lat) or np.isnan(lng)):
                    pts.append([lat, lng])
            except Exception:
                continue
    if centers_df is not None and hasattr(centers_df, "empty") and not centers_df.empty:
        try:
            pts += centers_df[["lat","lng"]].dropna().astype(float).values.tolist()
        except Exception:
            pass
    if not pts:
        return 36.2, 127.5, base_zoom
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    vlat = float(np.mean(arr[:, 0])); vlng = float(np.mean(arr[:, 1]))
    if arr.shape[0] > 1:
        span_lat = float(np.nanmax(arr[:, 0]) - np.nanmin(arr[:, 0]))
        span_lng = float(np.nanmax(arr[:, 1]) - np.nanmin(arr[:, 1]))
        span = max(span_lat, span_lng)
    else:
        span = 0.0
    zoom = 6.0 if span > 3.0 else base_zoom
    return vlat, vlng, zoom

def make_geojson_lines(items):
    feats=[]
    for it in (items or []):
        coords=it.get("path") or []
        if not isinstance(coords,list) or len(coords)<2: continue
        if any((pd.isna(x) or pd.isna(y)) for x,y in coords): continue
        feats.append({
            "type":"Feature",
            "properties":{"route": it.get("route",""),
                          "color": (it.get("color") or [28,200,138]) + [255],
                          "width": int(it.get("width") or 4)},
            "geometry":{"type":"LineString","coordinates": coords},
        })
    return {"type":"FeatureCollection","features":feats}

# ─────────────────────────────────────────────────────────────
# CSV 로딩
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_routes(src):
    df=pd.read_csv(src)
    need={"route","section","distance_km"}
    miss=need-set(df.columns)
    if miss: raise ValueError(f"routes.csv 필요 컬럼: {sorted(miss)}")
    df["route"]=df["route"].astype(str).str.strip().map(norm_name)
    df["section"]=df["section"].astype(str).str.strip()
    df["distance_km"]=pd.to_numeric(df["distance_km"],errors="coerce")
    if "id" not in df.columns:
        df["id"]=(df["route"].astype(str)+"@"+df["section"].astype(str)).str.replace(r"\s+","",regex=True)
    df["big"]=df["route"].map(ROUTE_TO_BIG).fillna("기타")
    df["big"]=pd.Categorical(df["big"],categories=TOP_ORDER,ordered=True)
    if "path" in df.columns:
        m=df["path"].notna()
        df.loc[m,"path"]=df.loc[m,"path"].map(parse_path)
    return df

@st.cache_data
def load_centers(src):
    if src is None: return None
    df=pd.read_csv(src)
    need={"route","center","address","lat","lng","id","seq"}
    miss=need-set(df.columns)
    if miss: raise ValueError(f"centers.csv 필요 컬럼: {sorted(miss)}")
    df["route"]=df["route"].astype(str).str.strip().map(norm_name)
    for c in ["center","address","id"]:
        df[c]=df[c].astype(str).str.strip()
    for c in ["lat","lng","seq","leg_km"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df["big"]=df["route"].map(ROUTE_TO_BIG).fillna("기타")
    df["big"]=pd.Categorical(df["big"],categories=TOP_ORDER,ordered=True)
    return df

# ─────────────────────────────────────────────────────────────
# 데이터/옵션
# ─────────────────────────────────────────────────────────────
st.sidebar.header("데이터")
use_repo=st.sidebar.radio("불러오기 방식",["Repo 내 파일","CSV 업로드"],index=0)
show_debug=st.sidebar.checkbox("디버그 보기", value=False)
if st.sidebar.button("↻ 캐시 초기화", use_container_width=True):
    st.cache_data.clear(); st.rerun()

if use_repo=="Repo 내 파일":
    routes=load_routes(Path("data/routes.csv"))
    centers=load_centers(Path("data/centers.csv") if Path("data/centers.csv").exists() else None)
else:
    r_up=st.sidebar.file_uploader("routes.csv 업로드", type=["csv"], key="routes_up")
    c_up=st.sidebar.file_uploader("centers.csv 업로드(선택)", type=["csv"], key="centers_up")
    if r_up is None:
        st.info("routes.csv를 올리면 시작합니다."); st.stop()
    routes=load_routes(r_up)
    centers=load_centers(c_up) if c_up else None

# 상태
st.session_state.setdefault("done_ids", set())

ROUTE_COLORS = {
    "아라자전거길": [0,173,181],
    "한강종주자전거길(서울구간)": [0,122,255],
    "남한강자전거길": [88,86,214],
    "새재자전거길": [255,159,10],
    "낙동강자전거길": [255,45,85],
    "금강자전거길": [255,204,0],
    "영산강자전거길": [52,199,89],
    "북한강자전거길": [142,142,147],
    "섬진강자전거길": [175,82,222],
    "오천자전거길": [255,55,95],
    "동해안자전거길(강원구간)": [90,200,250],
    "동해안자전거길(경북구간)": [0,199,190],
    "제주환상": [255,69,0],
}

# ─────────────────────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────────────────────
tab=st.radio("",["🚴 구간(거리) 추적","📍 인증센터"], horizontal=True, label_visibility="collapsed")

def pick_by_big(all_routes: list[str], key_prefix: str, use_defined=True, default_selected=None):
    big=st.sidebar.selectbox("대분류", TOP_ORDER, index=0, key=f"{key_prefix}_big")
    defined=[norm_name(r) for r in BIG_TO_ROUTES.get(big,[])] if use_defined else all_routes
    present=[r for r in defined if r in all_routes]
    absent=[r for r in defined if r not in all_routes]
    options=present+[r for r in absent if r in ALL_DEFINED_ROUTES]
    fmt=lambda r: r if r in present else f"{r}  • 데이터없음(폴백)"
    default = default_selected if default_selected is not None else []
    picked=st.sidebar.multiselect("노선(복수 선택 가능)", options, default=default,
                                  format_func=fmt, key=f"{key_prefix}_routes")
    return big, [norm_name(r) for r in picked]

def get_path_for_route(rname: str, ds_routes: pd.DataFrame, ds_centers: pd.DataFrame|None):
    # routes.path
    sub = ds_routes[ds_routes["route"] == rname]
    if not sub.empty and sub["path"].notna().any():
        p = sub["path"].dropna().iloc[0]
        if p and len(p) >= 2:
            return p
    # centers
    if ds_centers is not None:
        g = ds_centers[(ds_centers["route"]==rname)].dropna(subset=["lat","lng"]).sort_values("seq")
        if not g.empty:
            return g[["lng","lat"]].to_numpy(float).tolist()
    # fallback
    return FALLBACK_PATHS.get(rname)

# ─────────────────────────────────────────────────────────────
# 1) 구간(거리) 추적
# ─────────────────────────────────────────────────────────────
if tab=="🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")
    all_route_names=sorted(routes["route"].unique().tolist())
    big, picked = pick_by_big(all_route_names + ALL_DEFINED_ROUTES, "seg", use_defined=True, default_selected=[])

    # 노선 표(참고용)
    base = routes[routes["route"].isin(picked)][["route","section","distance_km","id"]].copy()
    base_view = base[["route","section","distance_km"]].rename(columns={
        "route":"route","section":"section","distance_km":"distance_km"
    })
    st.dataframe(base_view, use_container_width=True, hide_index=True)

    # 메트릭 계산
    sel_total = float(base["distance_km"].fillna(0).sum()) if not base.empty else \
                float(sum(OFFICIAL_TOTALS.get(r,0.0) for r in picked))
    # 빠른 체크(1클릭) – 라우트 단위
    st.sidebar.subheader("빠른 체크(1클릭)")
    quick_state={}
    for r in picked:
        # 현재 완료 판단: 세션의 done_ids(행 id 기반) 중 해당 라우트 id 하나라도 있으면 True
        ids_for_r = set(base.loc[base["route"]==r, "id"])
        cur = len(st.session_state.done_ids & ids_for_r) > 0
        quick_state[r] = st.sidebar.checkbox(r, value=cur, key=f"quick_{r}")

    # 사이드바 체크 → 세션 done_ids 갱신
    new_done_ids=set()
    for r, on in quick_state.items():
        if on:
            new_done_ids |= set(base.loc[base["route"]==r, "id"])
    st.session_state.done_ids = new_done_ids

    # 완료/남은 거리
    done_km = float(base.loc[base["id"].isin(st.session_state.done_ids),"distance_km"].fillna(0).sum())
    left_km = max(sel_total - done_km, 0.0)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{sel_total:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("대분류", big)

    # 지도 레이어: 베이스(회색)=선택 노선, 오버레이(색)=완료 체크 노선
    baseline_items=[]; overlay_items=[]; view_paths=[]
    for r in picked:
        path = get_path_for_route(r, routes, centers)
        if path and len(path)>=2:
            view_paths.append(path)
            baseline_items.append({"route": r, "path": path, "color": [190,190,190], "width": 5})
            # 완료면 컬러
            if quick_state.get(r, False):
                overlay_items.append({"route": r, "path": path, "color": ROUTE_COLORS.get(r,[28,200,138]), "width": 6})

    gj_base = make_geojson_lines(baseline_items)
    gj_done = make_geojson_lines(overlay_items)

    layers=[]
    if gj_base["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_base, pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=5))
    if gj_done["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_done, pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=6))

    vlat, vlng, vzoom = view_from_safe(view_paths, None, base_zoom=7.0 if len(picked)==1 else 5.8)
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text":"{properties.route}"},
    ), use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 2) 인증센터 (선택 노선만 회색, 완료센터(체크) 기준 구간을 초록/빨강)
# ─────────────────────────────────────────────────────────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다."); st.stop()

    st.sidebar.header("인증센터 필터")
    _, picked = pick_by_big(sorted(set(routes["route"])|set(centers["route"])|set(ALL_DEFINED_ROUTES)), "cent", use_defined=True, default_selected=[])

    dfc=centers[centers["route"].isin(picked)].copy()
    dfc=dfc.sort_values(["route","seq","center"]).reset_index(drop=True)

    # 빠른 체크(센터) – 센터 id 단위
    st.sidebar.subheader("센터 완료 체크(1클릭)")
    done_center_ids = set()
    for i,row in dfc.iterrows():
        cid=row["id"]
        cur = cid in st.session_state.done_ids
        checked = st.sidebar.checkbox(f'{row["route"]} - {row["center"]}', value=cur, key=f"cent_{cid}")
        if checked:
            done_center_ids.add(cid)
    st.session_state.done_ids = done_center_ids

    # 센터 간 구간 만들기
    seg=[]
    for r,g in dfc.groupby("route"):
        g=g.dropna(subset=["lat","lng"]).sort_values("seq")
        rec=g.to_dict("records")
        for i in range(len(rec)-1):
            a,b=rec[i],rec[i+1]
            path=[[float(a["lng"]),float(a["lat"])],[float(b["lng"]),float(b["lat"])]]
            done = (a["id"] in done_center_ids) and (b["id"] in done_center_ids)
            color = [28,200,138] if done else [230,57,70]
            seg.append({"route":r,"path":path,"color":color,"width":4,"done":done,
                        "start":a["center"],"end":b["center"]})

    seg_df=pd.DataFrame(seg)
    total=float(seg_df.shape[0])  # 개수 기준 표시(거리 컬럼이 없을 수도 있어 단순화)
    done=float(seg_df["done"].sum()) if not seg_df.empty else 0.0
    left=max(total-done,0.0)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("선택 인증센터 수", f"{dfc.shape[0]:,}")
    c2.metric("완료한 인증센터", f"{int(done):,}")
    c3.metric("완료 구간 수", f"{int(done):,}")
    c4.metric("미완료 구간 수", f"{int(left):,}")

    # 지도: 선택 노선 베이스(회색, 전체 윤곽) + 센터 구간(완료/미완료 색)
    baseline_items=[]; view_paths=[]
    for r in picked:
        path = get_path_for_route(r, routes, centers)
        if path and len(path)>=2:
            view_paths.append(path)
            baseline_items.append({"route": r, "path": path, "color": [190,190,190], "width": 5})

    gj_base = make_geojson_lines(baseline_items)
    gj_seg  = make_geojson_lines([{"route": s["route"], "path": s["path"], "color": s["color"], "width": s["width"]} for s in seg])

    layers=[]
    if gj_base["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_base, pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=5))
    if gj_seg["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_seg, pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=4))

    vlat, vlng, vzoom = view_from_safe(view_paths, None, 7.0)
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text":"{properties.route}"},
    ), use_container_width=True)
