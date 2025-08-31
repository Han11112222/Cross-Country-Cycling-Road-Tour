from __future__ import annotations
import json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

BUILD_TAG = "2025-09-02-v18-click1-grand"

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")
st.caption(f"BUILD: {BUILD_TAG}")

# ------------------------- 기본 데이터 -------------------------
OFFICIAL_TOTALS = {
    "아라자전거길": 21, "한강종주자전거길(서울구간)": 40, "남한강자전거길": 132, "새재자전거길": 100,
    "낙동강자전거길": 389, "금강자전거길": 146, "영산강자전거길": 133, "북한강자전거길": 70,
    "섬진강자전거길": 148, "오천자전거길": 105,
    "동해안자전거길(강원구간)": 242, "동해안자전거길(경북구간)": 76,
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
ALL_DEFINED_ROUTES=sorted({norm_name(r) for rs in BIG_TO_ROUTES.values() for r in rs})

# 폴백 경로(간략)
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
FALLBACK_PATHS={norm_name(k):v for k,v in _raw_fb.items()}

# ------------------------- 유틸 -------------------------
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

def view_from_safe(paths, centers_df, base_zoom:float):
    pts=[]
    for p in (paths or []):
        for xy in (p or []):
            try:
                lng,lat=float(xy[0]),float(xy[1])
                if not (np.isnan(lat) or np.isnan(lng)): pts.append([lat,lng])
            except: pass
    if centers_df is not None and hasattr(centers_df,"empty") and not centers_df.empty:
        try: pts+=centers_df[["lat","lng"]].dropna().astype(float).values.tolist()
        except: pass
    if not pts: return 36.2,127.5,base_zoom
    arr=np.asarray(pts,float)
    vlat=float(np.nanmean(arr[:,0])); vlng=float(np.nanmean(arr[:,1]))
    span=max(float(np.nanmax(arr[:,0])-np.nanmin(arr[:,0])),
             float(np.nanmax(arr[:,1])-np.nanmin(arr[:,1])))
    zoom=6.0 if span>3 else base_zoom
    return vlat,vlng,zoom

def make_geojson_lines(items):
    feats=[]
    for it in (items or []):
        coords=it.get("path") or []
        if not isinstance(coords,list) or len(coords)<2: continue
        try:
            if any((pd.isna(x) or pd.isna(y)) for x,y in coords): continue
        except: continue
        feats.append({
            "type":"Feature",
            "properties":{
                "route":it.get("route",""),
                "color":(it.get("color") or [180,180,180])+[255],
                "width":int(it.get("width") or 4)
            },
            "geometry":{"type":"LineString","coordinates":coords},
        })
    return {"type":"FeatureCollection","features":feats}

# ------------------------- 로딩 -------------------------
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
    for c in ["center","address","id"]:
        df[c]=df[c].astype(str).str.strip()
    for c in ["lat","lng","seq","leg_km"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    if auto_geo:
        need_geo=df[df["address"].notna()&(df["lat"].isna()|df["lng"].isna())]
        for i,row in need_geo.iterrows():
            try:
                r=requests.get("https://nominatim.openstreetmap.org/search",
                               params={"q":row["address"],"format":"json","limit":1},
                               headers={"User-Agent":"ccct/1.0"},timeout=10)
                if r.ok and r.json():
                    j=r.json()[0]
                    df.at[i,"lat"],df.at[i,"lng"]=float(j["lat"]),float(j["lon"])
                    time.sleep(1.0)
            except: pass
    return df

# ------------------------- 사이드바 -------------------------
st.sidebar.header("데이터")
use_repo=st.sidebar.radio("불러오기 방식",["Repo 내 파일","CSV 업로드"],index=0)
auto_geo=st.sidebar.toggle("주소 → 좌표 자동보정(지오코딩)", value=True)
if st.sidebar.button("↻ 캐시 초기화", use_container_width=True):
    st.cache_data.clear(); st.rerun()

if use_repo=="Repo 내 파일":
    routes=load_routes(Path("data/routes.csv"))
    centers=load_centers(Path("data/centers.csv"), auto_geo) if Path("data/centers.csv").exists() else None
else:
    r_up=st.sidebar.file_uploader("routes.csv 업로드", type=["csv"], key="routes_up")
    c_up=st.sidebar.file_uploader("centers.csv 업로드(선택)", type=["csv"], key="centers_up")
    if r_up is None: st.info("routes.csv를 올리면 시작합니다."); st.stop()
    routes=load_routes(r_up)
    centers=load_centers(c_up, auto_geo) if c_up else None

st.session_state.setdefault("done_routes", set())
st.session_state.setdefault("done_center_ids", set())

# ------------------------- 탭 -------------------------
tab=st.radio("",["🚴 구간(거리) 추적","📍 인증센터"], horizontal=True, label_visibility="collapsed")

# =================================================================
# 1) 거리 탭 — 회색 베이스 + '체크된 노선'만 흰색 오버레이
# =================================================================
if tab=="🚴 구간(거리) 추적":
    # 대분류/노선 선택
    st.sidebar.header("구간 선택")
    big = st.sidebar.selectbox("대분류", TOP_ORDER, index=0, key="seg_big")

    if big=="그랜드슬램":
        picked = BIG_TO_ROUTES["그랜드슬램"][:]   # 강제: 5개 모두
    else:
        all_routes=sorted(routes["route"].unique().tolist() + [r for r in ALL_DEFINED_ROUTES if r not in routes["route"].unique()])
        default_list = [r for r in BIG_TO_ROUTES.get(big, []) if r in all_routes] or all_routes[:1]
        picked = st.sidebar.multiselect("노선(복수 선택 가능)", all_routes, default=default_list, key="seg_routes")

    picked=[norm_name(r) for r in picked]

    # 각 노선별 '최적 경로' (routes.path > centers > fallback)
    def best_path(rname:str):
        p=None; src=""; pts=0
        sub=routes[routes["route"]==rname]
        if not sub.empty and sub["path"].notna().any():
            p=sub["path"].dropna().iloc[0]; src="routes.path"; pts=len(p)
        elif centers is not None:
            g=centers[(centers["route"]==rname)].dropna(subset=["lat","lng"]).sort_values("seq")
            if not g.empty:
                p=g[["lng","lat"]].astype(float).values.tolist(); src="centers"; pts=len(p)
        if p is None:
            fb=FALLBACK_PATHS.get(rname)
            if fb and len(fb)>=2: p=fb; src="fallback"; pts=len(fb)
        return p, src, pts

    # ✅ 체크박스: '완료 노선'만 저장 (1클릭 즉시 반영)
    st.subheader("완료 체크")
    done_routes=set()
    cols=[]
    for i,r in enumerate(picked):
        label=f"{r} — 전체구간 ({OFFICIAL_TOTALS.get(r,0)} km)"
        checked = st.checkbox(label, value=(r in st.session_state.done_routes), key=f"done_{r}")
        if checked: done_routes.add(r)
    st.session_state.done_routes = done_routes

    # 집계/경로 준비
    summary=[]; view_paths=[]; grey_rows=[]; white_rows=[]
    for r in picked:
        path, src, pts = best_path(r)
        disp_km=float(OFFICIAL_TOTALS.get(r,0.0))
        sub=routes[routes["route"]==r]
        sub_km=float(sub["distance_km"].fillna(0).sum()) if not sub.empty else 0.0
        if sub_km>0: disp_km=sub_km

        summary.append({"route":r,"경로소스":src,"포인트수":pts,"표시거리(km)":disp_km})

        if path and len(path)>=2:
            view_paths.append(path)
            # 회색 베이스(항상)
            grey_rows.append({"route":r,"path":path,"color":[185,185,185],"width":4})
            # 완료 체크된 노선만 흰색 오버레이
            if r in done_routes:
                white_rows.append({"route":r,"path":path,"color":[255,255,255],"width":7})

    with st.expander("선택 노선 총거리 요약", expanded=True):
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    # 메트릭(거리)
    total_km=sum(OFFICIAL_TOTALS.get(r,0.0) for r in picked)
    # routes.csv 존재 시 그 값 우선
    rt_sum=pd.DataFrame(summary)
    if not rt_sum.empty:
        total_km=float(rt_sum["표시거리(km)"].sum())
    done_km=sum(OFFICIAL_TOTALS.get(r,0.0) for r in done_routes)
    # routes.csv 기준으로 보정
    if not rt_sum.empty:
        done_km=float(rt_sum[rt_sum["route"].isin(done_routes)]["표시거리(km)"].sum())
    left_km=max(total_km-done_km,0.0)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("대분류", big)

    # 레이어 (회색 베이스 + 흰색 오버레이) — 기본 컬러(빨강) 없음
    gj_grey=make_geojson_lines(grey_rows)
    gj_white=make_geojson_lines(white_rows)
    layers=[]
    if gj_grey["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_grey,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=4, pickable=True))
    if gj_white["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_white,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=7, pickable=True))

    vlat,vlng,vzoom=view_from_safe(view_paths, None, base_zoom=7.0 if len(picked)==1 else 5.8)
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text":"{properties.route}"},
    ), use_container_width=True)

# =================================================================
# 2) 인증센터 탭 (기존 로직 유지)
# =================================================================
else:
    centers = centers
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다."); st.stop()

    st.sidebar.header("인증센터 필터")
    all_routes=sorted(set(centers["route"].map(norm_name)))
    picked=st.sidebar.multiselect("노선", all_routes,
                                  default=[r for r in BIG_TO_ROUTES.get("국토종주", []) if r in all_routes],
                                  key="cent_routes")

    dfc=centers[centers["route"].isin(picked)].copy()
    dfc=dfc.sort_values(["route","seq","center"]).reset_index(drop=True)
    done_centers=set(st.session_state.get("done_center_ids", set()))
    st.subheader("인증센터 완료 체크")
    new_done=set()
    for i,row in dfc.iterrows():
        cid=row["id"]; label=f"{row['route']} • {int(row['seq']) if not pd.isna(row['seq']) else ''}  {row['center']}"
        checked=st.checkbox(label, value=(cid in done_centers), key=f"cent_{cid}")
        if checked: new_done.add(cid)
    st.session_state.done_center_ids=new_done
    dfc["완료"]=dfc["id"].isin(new_done)

    # 센터 구간 선
    seg=[]
    for r,g in dfc.groupby("route"):
        g=g.sort_values("seq"); rec=g.to_dict("records")
        for i in range(len(rec)-1):
            a,b=rec[i],rec[i+1]
            if pd.isna(a.get("lat")) or pd.isna(a.get("lng")) or pd.isna(b.get("lat")) or pd.isna(b.get("lng")):
                continue
            dist=float(a.get("leg_km")) if not pd.isna(a.get("leg_km")) else (haversine_km(a.get("lat"),a.get("lng"),b.get("lat"),b.get("lng")) or 0.0)
            seg.append({"route":r,
                        "path":[[float(a["lng"]),float(a["lat"])],[float(b["lng"]),float(b["lat"])]],
                        "distance_km":0.0 if pd.isna(dist) else float(dist),
                        "done":bool(a["완료"] and b["완료"])})
    seg_df=pd.DataFrame(seg)
    total=float(seg_df["distance_km"].sum()) if not seg_df.empty else 0.0
    done=float(seg_df.loc[seg_df["done"],"distance_km"].sum()) if not seg_df.empty else 0.0
    left=max(total-done,0.0)

    c1,c2,c3=st.columns(3)
    c1.metric("센터 기준 총거리", f"{total:,.1f} km")
    c2.metric("완료", f"{done:,.1f} km")
    c3.metric("남은 거리", f"{left:,.1f} km")

    done_items=[{"route": r["route"], "path": r["path"], "color": [28,200,138], "width": 4}
                for _, r in seg_df[seg_df["done"]].iterrows()]
    todo_items=[{"route": r["route"], "path": r["path"], "color": [230,57,70], "width": 4}
                for _, r in seg_df[~seg_df["done"]].iterrows()]
    gj_done=make_geojson_lines(done_items)
    gj_todo=make_geojson_lines(todo_items)

    layers=[]
    if gj_todo["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_todo,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=4, pickable=True))
    if gj_done["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_done,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=4, pickable=True))

    geo=dfc.dropna(subset=["lat","lng"]).copy()
    if not geo.empty:
        geo["__color"]=geo["완료"].map(lambda b:[28,200,138] if b else [230,57,70])
        layers.append(pdk.Layer("ScatterplotLayer",
                                geo.rename(columns={"lat":"latitude","lng":"longitude"}),
                                get_position='[longitude, latitude]',
                                get_fill_color="__color", get_radius=160, pickable=True))

    vlat,vlng,vzoom=view_from_safe([], geo, 7.0)
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text":"{properties.route}"},
    ), use_container_width=True)
