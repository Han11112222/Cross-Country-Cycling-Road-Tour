# app.py — v14: 선택 없음=회색 베이스, 선택=하늘색 하이라이트(즉시 반영)
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

BUILD_TAG = "2025-08-31-v14"
st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")
st.caption(f"BUILD: {BUILD_TAG}")

# -------------------- 기본 데이터 --------------------
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
def norm(s:str)->str:
    s=str(s).strip()
    return "제주환상" if s=="제주환상자전거길" else s

# 폴백 경로([lng,lat]) — 최소한의 선 보장을 위해 사용
_raw_fb = {
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
FALLBACK_PATHS = {norm(k): v for k,v in _raw_fb.items()}

def parse_path(s):
    try:
        v=json.loads(s)
        if isinstance(v,list): return v
    except: pass
    return None

def haversine_km(a,b,c,d):
    R=6371.0088
    p1,p2=np.radians(a),np.radians(c)
    dphi,dlambda=np.radians(c-a),np.radians(d-b)
    x=np.sin(dphi/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dlambda/2)**2
    return float(R*2*np.arctan2(np.sqrt(x),np.sqrt(1-x)))

def densify(path, seg_km=5.0):
    if not path or len(path)<2: return path
    out=[path[0]]
    for (x1,y1),(x2,y2) in zip(path[:-1], path[1:]):
        d=haversine_km(y1,x1,y2,x2)
        n=max(1,int(d//seg_km))
        for i in range(1,n+1):
            t=i/(n+1)
            out.append([x1+(x2-x1)*t, y1+(y2-y1)*t])
        out.append([x2,y2])
    # 인접 중복 제거
    ded=[out[0]]
    for x,y in out[1:]:
        if x!=ded[-1][0] or y!=ded[-1][1]:
            ded.append([x,y])
    return ded

def make_geojson(items):
    feats=[]
    for it in items:
        coords=it["path"]
        feats.append({
            "type":"Feature",
            "properties":{"route":it["route"],"color":it["color"]+[255],"width":it["width"]},
            "geometry":{"type":"LineString","coordinates":coords},
        })
    return {"type":"FeatureCollection","features":feats}

def view_from(paths, base_zoom=6.0):
    pts=[]
    for p in (paths or []):
        for x,y in p: pts.append([y,x])  # lat,lng
    if not pts: return 36.2,127.5,base_zoom
    arr=np.array(pts,float)
    vlat,vlng=arr[:,0].mean(),arr[:,1].mean()
    span=max(arr[:,0].ptp(),arr[:,1].ptp())
    return float(vlat), float(vlng), (6.0 if span>3 else base_zoom)

# -------------------- 로딩 --------------------
@st.cache_data
def load_routes(src):
    df=pd.read_csv(src)
    need={"route","section","distance_km"}
    miss=need-set(df.columns)
    if miss: raise ValueError(f"routes.csv 필요 컬럼: {sorted(miss)}")
    df["route"]=df["route"].astype(str).str.strip().map(norm)
    df["section"]=df["section"].astype(str).str.strip()
    df["distance_km"]=pd.to_numeric(df["distance_km"], errors="coerce")
    if "path" in df.columns:
        m=df["path"].notna()
        df.loc[m,"path"]=df.loc[m,"path"].map(parse_path)
    return df

st.sidebar.header("데이터")
use_repo=st.sidebar.radio("불러오기",["Repo 파일","CSV 업로드"],index=0)
if st.sidebar.button("↻ 캐시 초기화", use_container_width=True):
    st.cache_data.clear(); st.rerun()

if use_repo=="Repo 파일":
    routes=load_routes(Path("data/routes.csv"))
else:
    up=st.sidebar.file_uploader("routes.csv 업로드", type=["csv"])
    if not up: st.stop()
    routes=load_routes(up)

# -------------------- 노선→경로 만들기 --------------------
route_names=sorted(routes["route"].unique().tolist())
items=[]
for r in route_names:
    p=None
    sub=routes[routes["route"]==r]
    if "path" in sub.columns and sub["path"].notna().any():
        p=sub["path"].dropna().iloc[0]
    if p is None: p=FALLBACK_PATHS.get(r)
    if p and len(p)>=2:
        items.append({"route":r, "path":densify(p,5.0), "km":float(OFFICIAL_TOTALS.get(r,0))})

# -------------------- UI: 하이라이트 선택 --------------------
st.subheader("하이라이트 선택(여러 개 가능)")
highlight = st.multiselect("지도로 강조할 노선", [it["route"] for it in items], default=[])

BASE_COLOR=[170,170,170]; BASE_W=3
HL_COLOR=[0,200,255]; HL_W=7

# 요약
total_km=sum(it["km"] for it in items)
done_km=sum(it["km"] for it in items if it["route"] in highlight)
left_km=max(total_km-done_km,0.0)
c1,c2,c3,c4=st.columns(4)
c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
c2.metric("완료 누적거리(하이라이트 합)", f"{done_km:,.1f} km")
c3.metric("남은 거리", f"{left_km:,.1f} km")
c4.metric("대분류", "국토종주")

# 지도 레이어
# 1) 전체 베이스(회색)
base_items=[{"route":it["route"], "path":it["path"], "color":BASE_COLOR, "width":BASE_W} for it in items]
gj_base=make_geojson(base_items)
layers=[pdk.Layer("GeoJsonLayer", gj_base,
                  get_line_color="properties.color",
                  get_line_width="properties.width",
                  line_width_min_pixels=3, pickable=False)]
# 2) 하이라이트(하늘색)
hi=[it for it in items if it["route"] in highlight]
if hi:
    gj_hi=make_geojson([{"route":it["route"],"path":it["path"],"color":HL_COLOR,"width":HL_W} for it in hi])
    layers.append(pdk.Layer("GeoJsonLayer", gj_hi,
                            get_line_color="properties.color",
                            get_line_width="properties.width",
                            line_width_min_pixels=6, pickable=True))

# 화면 위치
vlat,vlng,vzoom=view_from([it["path"] for it in items], base_zoom=5.8)
st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
    tooltip={"text":"{properties.route}"},
), use_container_width=True)
