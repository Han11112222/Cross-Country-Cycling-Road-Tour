# app.py — v16: 선택 없음=회색 베이스, 선택(완료)=노선색 덮어쓰기, 즉시 반영(1클릭)
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

BUILD_TAG = "2025-09-01-v16"

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")
st.caption(f"BUILD: {BUILD_TAG}")

# ───────────────── 공인 총거리 ─────────────────
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

# ─────────────── 분류/명칭 표준화 ───────────────
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

# ─────────────── 폴백 경로([lng,lat]) ───────────────
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

# ─────────────── 유틸 ───────────────
def parse_path(s):
    try:
        v=json.loads(s)
        if isinstance(v,list): return v
    except Exception: pass
    return None

@st.cache_data(ttl=60*60*24)
def geocode(addr:str):
    try:
        r=requests.get("https://nominatim.openstreetmap.org/search",
                       params={"q":addr,"format":"json","limit":1},
                       headers={"User-Agent":"ccct/1.0"}, timeout=10)
        if r.ok and r.json():
            j=r.json()[0]; return float(j["lat"]), float(j["lon"])
    except Exception:
        pass
    return None,None

def view_from_safe(paths, centers_df, base_zoom: float):
    pts=[]
    for p in (paths or []):
        for xy in (p or []):
            try:
                lng, lat = float(xy[0]), float(xy[1])
                if not (np.isnan(lat) or np.isnan(lng)):
                    pts.append([lat, lng])
            except Exception:
                continue
    if centers_df is not None and hasattr(centers_df,"empty") and not centers_df.empty:
        try:
            pts += centers_df[["lat","lng"]].dropna().astype(float).values.tolist()
        except Exception:
            pass
    if not pts:
        return 36.2, 127.5, base_zoom
    arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    vlat = float(np.mean(arr[:,0])); vlng = float(np.mean(arr[:,1]))
    if arr.shape[0] > 1:
        span_lat = float(np.nanmax(arr[:,0]) - np.nanmin(arr[:,0]))
        span_lng = float(np.nanmax(arr[:,1]) - np.nanmin(arr[:,1]))
        span = max(span_lat, span_lng)
    else:
        span = 0.0
    zoom = 6.0 if span > 3.0 else base_zoom
    return vlat, vlng, zoom

def make_geojson_lines(items):
    feats=[]
    for it in (items or []):
        coords=it.get("path") or []
        if not isinstance(coords,list) or len(coords)<2: 
            continue
        try:
            bad = any((pd.isna(x) or pd.isna(y)) for x,y in coords)
        except Exception:
            bad = False
        if bad: 
            continue
        feats.append({
            "type":"Feature",
            "properties":{
                "route": it.get("route",""),
                "color": (it.get("color") or [200,200,200]) + [255],
                "width": int(it.get("width") or 5),
            },
            "geometry":{"type":"LineString","coordinates": coords},
        })
    return {"type":"FeatureCollection","features":feats}

# ─────────────── 로딩 ───────────────
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
def load_centers(src, auto_geo: bool):
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
    if auto_geo:
        needs=df[df["address"].notna() & (df["lat"].isna() | df["lng"].isna())]
        for i,row in needs.iterrows():
            lat,lng=geocode(row["address"])
            if lat is not None and lng is not None:
                df.at[i,"lat"], df.at[i,"lng"]=lat,lng
                time.sleep(1.0)
    df["big"]=df["route"].map(ROUTE_TO_BIG).fillna("기타")
    df["big"]=pd.Categorical(df["big"],categories=TOP_ORDER,ordered=True)
    return df

# ─────────────── 사이드바 ───────────────
st.sidebar.header("데이터")
use_repo=st.sidebar.radio("불러오기 방식",["Repo 내 파일","CSV 업로드"],index=0)
auto_geo=st.sidebar.toggle("주소 → 좌표 자동보정(지오코딩)", value=True)
show_debug=st.sidebar.checkbox("디버그 보기", value=False)
if st.sidebar.button("↻ 캐시 초기화", use_container_width=True):
    st.cache_data.clear(); st.rerun()

if use_repo=="Repo 내 파일":
    routes=load_routes(Path("data/routes.csv"))
    centers=load_centers(Path("data/centers.csv"), auto_geo) if Path("data/centers.csv").exists() else None
else:
    r_up=st.sidebar.file_uploader("routes.csv 업로드", type=["csv"], key="routes_up")
    c_up=st.sidebar.file_uploader("centers.csv 업로드(선택)", type=["csv"], key="centers_up")
    if r_up is None:
        st.info("routes.csv를 올리면 시작합니다."); st.stop()
    routes=load_routes(r_up)
    centers=load_centers(c_up, auto_geo) if c_up else None

st.session_state.setdefault("done_section_ids", set())

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

tab=st.radio("",["🚴 구간(거리) 추적","📍 인증센터"], horizontal=True, label_visibility="collapsed")

def pick_by_big(all_routes: list[str], key_prefix: str, use_defined=True):
    big=st.sidebar.selectbox("대분류", TOP_ORDER, index=0, key=f"{key_prefix}_big")
    defined=[norm_name(r) for r in BIG_TO_ROUTES.get(big,[])] if use_defined else all_routes
    present=[r for r in defined if r in all_routes]
    absent=[r for r in defined if r not in all_routes]
    options=present+[r for r in absent if r in ALL_DEFINED_ROUTES]
    fmt=lambda r: r if r in present else f"{r}  • 데이터없음(폴백)"
    picked=st.sidebar.multiselect("노선(복수 선택 가능)", options, default=present or options[:1],
                                  format_func=fmt, key=f"{key_prefix}_routes")
    return big, [norm_name(r) for r in picked]

# ─────────────── 1) 구간(거리) 추적 ───────────────
if tab=="🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")
    all_route_names=sorted(routes["route"].unique().tolist())
    big, picked = pick_by_big(all_route_names + ALL_DEFINED_ROUTES, "seg", use_defined=True)

    # path 확보: routes.path → centers → fallback
    routes2=routes.copy()
    if "path" in routes2.columns:
        m=routes2["path"].notna()
        routes2.loc[m,"path"]=routes2.loc[m,"path"].map(parse_path)

    def centers_path(rname:str):
        if centers is None: return None
        g=centers[(centers["route"]==rname)].dropna(subset=["lat","lng"]).sort_values("seq")
        if g.empty: return None
        return g[["lng","lat"]].to_numpy(float).tolist()

    # 요약 테이블(편집용)
    base=routes[routes["route"].isin(picked)][["route","section","distance_km","id"]].copy()
    base["완료"]=base["id"].isin(st.session_state.done_section_ids)

    edited=st.data_editor(
        base,  # id 컬럼 그대로 둠(매핑 꼬임 방지)
        use_container_width=True,
        hide_index=True,
        column_config={
            "완료": st.column_config.CheckboxColumn(label="완료", help="체크하면 지도에 색상이 입혀집니다.", default=False),
            "id": st.column_config.Column(label="id", width="small", disabled=True),
        },
        key="editor_routes",
    )

    # 클릭 1번으로 즉시 반영: 현재 표 상태 그대로 읽어 세트 갱신
    new_done=set(edited.loc[edited["완료"], "id"].astype(str).tolist())
    if new_done != st.session_state.done_section_ids:
        st.session_state.done_section_ids = new_done

    total_km=float(base["distance_km"].fillna(0).sum()) if not base.empty else 0.0
    done_km=float(base.loc[base["id"].isin(st.session_state.done_section_ids), "distance_km"].fillna(0).sum())
    left_km=max(total_km-done_km,0.0)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("대분류", big)

    # 지도 레이어 구성
    base_lines=[]   # 회색 전체 구간(선택한 노선)
    done_lines=[]   # 완료 체크된 노선(컬러)

    view_paths=[]

    for r in picked:
        color = ROUTE_COLORS.get(r, [28,200,138])
        # path 소스 선택
        p_row = routes2[routes2["route"]==r]
        path = None
        if not p_row.empty and p_row["path"].notna().any():
            path = p_row["path"].dropna().iloc[0]
        if not path:
            path = centers_path(r)
        if not path:
            path = FALLBACK_PATHS.get(r)

        if path and len(path)>=2:
            base_lines.append({"route": r, "path": path, "color": [185,185,185], "width": 6})
            view_paths.append(path)

        # 이 노선이 완료 체크되었으면(현재 구조상 section=전체구간 1행이므로 id로 판단)
        ids_of_r = set(base.loc[base["route"]==r, "id"].astype(str))
        if ids_of_r & st.session_state.done_section_ids:
            done_lines.append({"route": r, "path": path, "color": color, "width": 7})

    gj_base = make_geojson_lines(base_lines)
    gj_done = make_geojson_lines(done_lines)

    layers=[]
    if gj_base["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_base, pickable=False,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=4))
    if gj_done["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_done, pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=5))

    vlat, vlng, vzoom = view_from_safe(view_paths, None, base_zoom=7.0 if len(picked)==1 else 5.8)

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text":"{properties.route}"},
    ), use_container_width=True)

# ─────────────── 2) 인증센터(지도는 동일 패턴 유지, 필요시 확장) ───────────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다."); st.stop()

    _, picked = pick_by_big(sorted(set(routes["route"])|set(centers["route"])|set(ALL_DEFINED_ROUTES)), "cent", use_defined=True)
    dfc=centers[centers["route"].isin(picked)].copy().sort_values(["route","seq","center"]).reset_index(drop=True)

    st.dataframe(dfc[["route","seq","center","address","lat","lng"]], use_container_width=True, hide_index=True)

    # 센터 사이 연결선(회색 베이스만 간단히)
    base_lines=[]
    view_paths=[]
    for r,g in dfc.groupby("route"):
        g=g.dropna(subset=["lat","lng"]).sort_values("seq")
        if len(g)<2: 
            continue
        pts=g[["lng","lat"]].to_numpy(float).tolist()
        base_lines.append({"route": r, "path": pts, "color": [185,185,185], "width": 6})
        view_paths.append(pts)

    gj_base=make_geojson_lines(base_lines)
    layers=[]
    if gj_base["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_base,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=4))
    vlat, vlng, vzoom = view_from_safe(view_paths, None, 6.2)
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text":"{properties.route}"},
    ), use_container_width=True)

# 디버그 보조
if show_debug:
    st.write("done_ids:", sorted(st.session_state.done_section_ids))
