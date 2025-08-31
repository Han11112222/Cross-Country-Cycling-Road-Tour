# app.py — 선택 없음=회색 베이스만, 선택=하늘색 하이라이트(즉시 반영)
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

BUILD_TAG = "2025-08-31-geojson-v13"
st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")
st.caption(f"BUILD: {BUILD_TAG}")

# ───────────────── 기본 정의 ─────────────────
OFFICIAL_TOTALS = {
    "아라자전거길": 21, "한강종주자전거길(서울구간)": 40, "남한강자전거길": 132,
    "새재자전거길": 100, "낙동강자전거길": 389, "금강자전거길": 146,
    "영산강자전거길": 133, "북한강자전거길": 70, "섬진강자전거길": 148,
    "오천자전거길": 105, "동해안자전거길(강원구간)": 242, "동해안자전거길(경북구간)": 76,
    "제주환상": 234, "제주환상자전거길": 234,
}
TOP_ORDER = ["국토종주", "4대강 종주", "그랜드슬램", "제주환상"]
BIG_TO_ROUTES = {
    "국토종주": ["아라자전거길","한강종주자전거길(서울구간)","남한강자전거길","새재자전거길","낙동강자전거길"],
    "4대강 종주": ["한강종주자전거길(서울구간)","금강자전거길","영산강자전거길","낙동강자전거길"],
    "그랜드슬램": ["북한강자전거길","섬진강자전거길","오천자전거길","동해안자전거길(강원구간)","동해안자전거길(경북구간)"],
    "제주환상": ["제주환상","제주환상자전거길"],
}
def norm_name(s:str)->str:
    s=str(s).strip()
    return "제주환상" if s=="제주환상자전거길" else s
ROUTE_TO_BIG={norm_name(r):b for b,rs in BIG_TO_ROUTES.items() for r in rs}
ALL_DEFINED_ROUTES=sorted({norm_name(r) for v in BIG_TO_ROUTES.values() for r in v})

# 폴백 경로([lng,lat])
_raw_fb={
    "아라자전거길": [[126.58,37.60],[126.68,37.60],[126.82,37.57]],
    "한강종주자전거길(서울구간)": [[126.82,37.57],[127.02,37.55],[127.08,37.54]],
    "남한강자전거길": [[127.31,37.55],[127.63,37.29],[127.90,36.98]],
    "새재자전거길": [[127.90,36.98],[128.07,36.69],[128.16,36.41]],
    "낙동강자전거길": [[128.72,36.56],[128.60,35.87],[128.50,35.40],[129.03,35.10]],
    "금강자전거길": [[127.48,36.44],[127.28,36.50],[127.12,36.45],[126.71,36.00]],
    "영산강자전거길": [[126.99,35.32],[126.72,35.02],[126.39,34.79]],
    "북한강자전거길": [[127.31,37.55],[127.63,37.74],[127.73,37.88]],
    "섬진강자전거길": [[127.38,35.41],[127.47,35.22],[127.75,35.10],[127.69,34.94]],
    "오천자전거길": [[126.60,36.33],[126.85,36.40],[127.12,36.45]],
    "동해안자전거길(강원구간)": [[128.45,38.38],[128.60,38.20],[129.00,37.75],[129.20,37.44]],
    "동해안자전거길(경북구간)": [[129.20,37.44],[129.36,36.03],[129.31,35.84],[129.35,35.55]],
    "제주환상": [[126.32,33.50],[126.70,33.52],[126.95,33.45],[126.95,33.25],[126.60,33.23],[126.32,33.35],[126.32,33.50]],
}
FALLBACK_PATHS={norm_name(k):v for k,v in _raw_fb.items()}

def parse_path(s):
    try:
        v=json.loads(s)
        if isinstance(v,list): return v
    except Exception: pass
    return None

def haversine_km(a,b,c,d):
    if any(pd.isna([a,b,c,d])): return np.nan
    R=6371.0088
    p1,p2=math.radians(a),math.radians(c)
    dphi,dlambda=math.radians(c-a),math.radians(d-b)
    x=math.sin(dphi/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return R*2*math.atan2(math.sqrt(x),math.sqrt(1-x))

def densify_path(path, segment_km:float=5.0):
    if not path or len(path)<2: return path
    out=[path[0]]
    for (x1,y1),(x2,y2) in zip(path[:-1], path[1:]):
        d=haversine_km(y1,x1,y2,x2)
        if d is None or np.isnan(d): continue
        n=max(1,int(d//segment_km))
        for i in range(1,n+1):
            t=i/(n+1)
            out.append([x1+(x2-x1)*t, y1+(y2-y1)*t])
        out.append([x2,y2])
    # 인접 중복 제거
    dedup=[out[0]]
    for x,y in out[1:]:
        if x!=dedup[-1][0] or y!=dedup[-1][1]:
            dedup.append([x,y])
    return dedup

def make_geojson(items):
    feats=[]
    for it in (items or []):
        coords=it.get("path") or []
        if not isinstance(coords,list) or len(coords)<2: continue
        if any((pd.isna(x) or pd.isna(y)) for x,y in coords): continue
        feats.append({
            "type":"Feature",
            "properties":{"route":it.get("route",""),
                          "color":(it.get("color") or [28,200,255])+[255],
                          "width":int(it.get("width") or 6)},
            "geometry":{"type":"LineString","coordinates":coords},
        })
    return {"type":"FeatureCollection","features":feats}

def view_from(paths, centers_df, base_zoom:float):
    pts=[]
    for p in (paths or []):
        for xy in (p or []):
            try:
                lng,lat=float(xy[0]),float(xy[1])
                if not (np.isnan(lat) or np.isnan(lng)):
                    pts.append([lat,lng])
            except: pass
    if centers_df is not None and hasattr(centers_df,"empty") and not centers_df.empty:
        try: pts += centers_df[["lat","lng"]].dropna().astype(float).values.tolist()
        except: pass
    if not pts: return 36.2,127.5,base_zoom
    arr=np.asarray(pts,float)
    vlat,vlng=float(arr[:,0].mean()),float(arr[:,1].mean())
    span=max(np.ptp(arr[:,0]),np.ptp(arr[:,1])) if arr.shape[0]>1 else 0.0
    zoom=6.0 if span>3.0 else base_zoom
    return vlat,vlng,zoom

# ───────────────── 로딩 ─────────────────
@st.cache_data
def load_routes(src):
    df=pd.read_csv(src)
    need={"route","section","distance_km"}
    miss=need-set(df.columns)
    if miss: raise ValueError(f"routes.csv 필요 컬럼: {sorted(miss)}")
    df["route"]=df["route"].astype(str).str.strip().map(norm_name)
    df["section"]=df["section"].astype(str).str.strip()
    df["distance_km"]=pd.to_numeric(df["distance_km"],errors="coerce")
    if "path" in df.columns:
        m=df["path"].notna()
        df.loc[m,"path"]=df.loc[m,"path"].map(parse_path)
    return df

@st.cache_data
def load_centers(src, auto_geo:bool):
    if src is None: return None
    df=pd.read_csv(src)
    need={"route","center","address","lat","lng","id","seq"}
    miss=need-set(df.columns)
    if miss: raise ValueError(f"centers.csv 필요 컬럼: {sorted(miss)}")
    df["route"]=df["route"].astype(str).str.strip().map(norm_name)
    for c in ["center","address","id"]: df[c]=df[c].astype(str).str.strip()
    for c in ["lat","lng","seq","leg_km"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    if auto_geo:
        need=df[df["address"].notna() & (df["lat"].isna() | df["lng"].isna())]
        for i,row in need.iterrows():
            try:
                j=requests.get("https://nominatim.openstreetmap.org/search",
                               params={"q":row["address"],"format":"json","limit":1},
                               headers={"User-Agent":"ccct/1.0"},timeout=10).json()
                if j:
                    df.at[i,"lat"], df.at[i,"lng"]=float(j[0]["lat"]), float(j[0]["lon"])
                    time.sleep(1.0)
            except: pass
    return df

# ───────────────── 옵션 ─────────────────
st.sidebar.header("데이터")
use_repo=st.sidebar.radio("불러오기 방식",["Repo 내 파일","CSV 업로드"],index=0)
auto_geo=st.sidebar.toggle("주소 → 좌표 자동보정(지오코딩)", value=True)
show_baseline=st.sidebar.toggle("회색 베이스라인(전체 노선) 표시", value=True)
if st.sidebar.button("↻ 캐시 초기화", use_container_width=True):
    st.cache_data.clear(); st.rerun()

if use_repo=="Repo 내 파일":
    routes=load_routes(Path("data/routes.csv"))
    centers=load_centers(Path("data/centers.csv"), auto_geo) if Path("data/centers.csv").exists() else None
else:
    r_up=st.sidebar.file_uploader("routes.csv 업로드", type=["csv"])
    c_up=st.sidebar.file_uploader("centers.csv 업로드(선택)", type=["csv"])
    if r_up is None:
        st.info("routes.csv를 올리면 시작합니다."); st.stop()
    routes=load_routes(r_up)
    centers=load_centers(c_up, auto_geo) if c_up else None

GREY=[170,170,170]; GREY_W=3
HL=[0,200,255]; HL_W=7

tab=st.radio("",["🚴 구간(거리) 추적","📍 인증센터"], horizontal=True, label_visibility="collapsed")

def pick_by_big(all_routes:list[str], key_prefix:str, use_defined=True):
    big=st.sidebar.selectbox("대분류", TOP_ORDER, index=0, key=f"{key_prefix}_big")
    defined=[norm_name(r) for r in BIG_TO_ROUTES.get(big,[])] if use_defined else all_routes
    present=[r for r in defined if r in all_routes]
    absent=[r for r in defined if r not in all_routes]
    options=present+[r for r in absent if r in ALL_DEFINED_ROUTES]
    fmt=lambda r: r if r in present else f"{r}  • 데이터없음(폴백)"
    picked=st.sidebar.multiselect("노선(복수 선택 가능)", options, default=present or options[:1],
                                  format_func=fmt, key=f"{key_prefix}_routes")
    return big, [norm_name(r) for r in picked]

# ───────── 1) 구간(거리) 추적 ─────────
if tab=="🚴 구간(거리) 추적":
    all_route_names=sorted(routes["route"].unique().tolist())
    big, picked = pick_by_big(all_route_names + ALL_DEFINED_ROUTES, "seg", use_defined=True)

    # 노선별 path 구성: routes.path > centers > fallback
    def centers_path(rname:str):
        if centers is None: return None, np.nan
        g=centers[(centers["route"]==rname)].dropna(subset=["lat","lng"]).sort_values("seq")
        if g.empty: return None, np.nan
        pts=g[["lng","lat"]].to_numpy(float).tolist()
        km=float(g["leg_km"].fillna(0).sum()) if ("leg_km" in g.columns and g["leg_km"].notna().any()) \
           else sum(haversine_km(pts[i][1],pts[i][0],pts[i+1][1],pts[i+1][0]) for i in range(len(pts)-1))
        return pts, km

    items=[]; view_paths=[]
    for r in picked:
        path=None; km=float(OFFICIAL_TOTALS.get(r,0))
        sub=routes[routes["route"]==r]
        if not sub.empty and "path" in sub.columns and sub["path"].notna().any():
            p=sub["path"].dropna().iloc[0]
            if p and len(p)>=2: path=p
        if path is None:
            p2,k2=centers_path(r)
            if p2 and len(p2)>=2: path=p2; 
            if p2 and not np.isnan(k2): km=float(k2)
        if path is None:
            fb=FALLBACK_PATHS.get(r)
            if fb and len(fb)>=2: path=fb
        if path is not None:
            path=densify_path(path, 5.0)
            items.append({"route":norm_name(r), "path":path, "km":km})
            view_paths.append(path)

    # ✅ 하이라이트 멀티선택 (기본: 선택 없음)
    st.subheader("하이라이트 선택(완료 간주)")
    highlight = st.multiselect(
        "지도로 강조할 노선을 고르세요(여러 개 가능)",
        [it["route"] for it in items],
        default=[],  # ← 기본 비어있음
        key="highlight_routes"
    )
    highlight=set(map(norm_name, highlight))

    # 요약 카드
    total_km=sum(it["km"] for it in items)
    done_km=sum(it["km"] for it in items if it["route"] in highlight)
    left_km=max(total_km-done_km,0.0)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리(하이라이트 합)", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("대분류", big)

    # 지도 레이어
    layers=[]
    if items and show_baseline:
        grey_items=[{"route":it["route"], "path":it["path"], "color":GREY, "width":GREY_W} for it in items]
        gj_grey=make_geojson(grey_items)
        layers.append(pdk.Layer("GeoJsonLayer", gj_grey,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=3, pickable=False))
    hi_items=[{"route":it["route"], "path":it["path"], "color":HL, "width":HL_W}
              for it in items if it["route"] in highlight]
    if hi_items:
        gj_hi=make_geojson(hi_items)
        layers.append(pdk.Layer("GeoJsonLayer", gj_hi,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=6, pickable=True))

    # 센터 점(연한 회색, 참고용)
    centers_for_view=None
    if centers is not None:
        g=centers[centers["route"].isin(picked)].dropna(subset=["lat","lng"]).copy()
        if not g.empty:
            centers_for_view=g.copy()
            g["__color"]=[[210,210,210]]*len(g)
            layers.append(pdk.Layer("ScatterplotLayer",
                                    g.rename(columns={"lat":"latitude","lng":"longitude"}),
                                    get_position='[longitude, latitude]',
                                    get_fill_color="__color", get_radius=120, pickable=False))

    vlat,vlng,vzoom = view_from(view_paths, centers_for_view, base_zoom=7.0 if len(picked)==1 else 5.8)
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text":"{properties.route}"},
    ), use_container_width=True)

# ───────── 2) 인증센터(필요 시 동일 패턴으로 확장) ─────────
else:
    st.info("인증센터 탭은 필요 시 동일한 하이라이트 멀티선택 패턴을 적용할 수 있어요.")
