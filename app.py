# app.py — v22: 인증센터 폴백-보간 좌표 + 온더플라이 지오코딩 + 인덱스 안전 반영
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

BUILD_TAG = "2025-09-01-v22"
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
    "국토종주": ["아라자전거길","한강종주자전거길(서울구간)","남한강자전거길","새재자전거길","낙동강자전거길"],
    "4대강 종주": ["한강종주자전거길(서울구간)","금강자전거길","영산강자전거길","낙동강자전거길"],
    "제주환상": ["제주환상","제주환상자전거길"],
}
def norm_name(s: str) -> str:
    s = str(s).strip()
    return "제주환상" if s == "제주환상자전거길" else s

ALL_DEFINED_ROUTES = sorted({
    "아라자전거길","한강종주자전거길(서울구간)","남한강자전거길","새재자전거길","낙동강자전거길",
    "금강자전거길","영산강자전거길","북한강자전거길","섬진강자전거길","오천자전거길",
    "동해안자전거길(강원구간)","동해안자전거길(경북구간)","제주환상","제주환상자전거길"
})
ROUTE_TO_BIG = {norm_name(r): big for big, rs in BIG_TO_ROUTES.items() for r in rs}

# ─────────────────────────────────────────────────────────────
# 폴백 경로([lng,lat]) — 데이터 없을 때 회색 기준선 및 인증센터 임시보간에 사용
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
# 유틸/지오코딩/보간
# ─────────────────────────────────────────────────────────────
def haversine_km(a,b,c,d):
    if any(pd.isna([a,b,c,d])): return np.nan
    R=6371.0088
    p1,p2=math.radians(a),math.radians(c)
    dphi,dlambda=math.radians(c-a),math.radians(d-b)
    x=math.sin(dphi/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return R*2*math.atan2(math.sqrt(x),math.sqrt(1-x))

def parse_path(s):
    try:
        v=json.loads(s)
        if isinstance(v,list): return v
    except Exception:
        pass
    return None

@st.cache_data(ttl=60*60*24)
def geocode(addr:str):
    try:
        r=requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q":addr,"format":"json","limit":1},
            headers={"User-Agent":"ccct/1.0"},
            timeout=10,
        )
        if r.ok and r.json():
            j=r.json()[0]; return float(j["lat"]), float(j["lon"])
    except Exception:
        pass
    return None,None

def _append_path_points(pts, p):
    if not isinstance(p, (list, tuple)) or len(p)==0:
        return
    if len(p)==2 and all(isinstance(v,(int,float)) for v in p):
        candidates = [p]
    else:
        candidates = list(p)
    for xy in candidates:
        try:
            lng, lat = float(xy[0]), float(xy[1])
            if not (pd.isna(lat) or pd.isna(lng)):
                pts.append([lat, lng])
        except Exception:
            continue

def view_from(paths, centers_df, base_zoom: float):
    pts=[]
    for p in (paths or []):
        _append_path_points(pts, p)
    try:
        if centers_df is not None and hasattr(centers_df, "empty") and not centers_df.empty:
            for lat, lng in centers_df[["lat","lng"]].dropna().astype(float).values.tolist():
                if not (pd.isna(lat) or pd.isna(lng)):
                    pts.append([lat,lng])
    except Exception:
        pass
    if pts:
        arr=np.array(pts, dtype=float)
        if arr.ndim==1: arr=arr.reshape(1,-1)
        if arr.shape[1] != 2:
            return 36.2, 127.5, base_zoom
        vlat=float(arr[:,0].mean()); vlng=float(arr[:,1].mean())
        span = max(float(np.ptp(arr[:,0])) if arr.shape[0]>1 else 0.0,
                   float(np.ptp(arr[:,1])) if arr.shape[0]>1 else 0.0)
        zoom = 6.0 if span>3 else base_zoom
        return vlat, vlng, zoom
    return 36.2, 127.5, base_zoom

def make_geojson_lines(line_items):
    feats=[]
    for it in line_items:
        coords=it.get("path") or []
        if not isinstance(coords,list) or len(coords)<2: continue
        if any((pd.isna(x) or pd.isna(y)) for x,y in coords): continue
        feats.append({
            "type":"Feature",
            "properties":{
                "route": it.get("route",""),
                "color": (it.get("color") or [28,200,138]) + [255],
                "width": int(it.get("width") or 4),
            },
            "geometry":{"type":"LineString","coordinates": coords},
        })
    return {"type":"FeatureCollection","features":feats}

def _interp_on_polyline(path_lnglat, t: float):
    """path: [[lng,lat],...] / t in [0,1] → (lat,lng) 보간"""
    if not path_lnglat or len(path_lnglat)<2: return None,None
    # 누적 길이
    cum=[0.0]
    for i in range(len(path_lnglat)-1):
        a,b=path_lnglat[i],path_lnglat[i+1]
        cum.append(cum[-1]+(haversine_km(a[1],a[0],b[1],b[0]) or 0.0))
    L=cum[-1] if cum[-1]>0 else 1.0
    s=t*L
    j=0
    while j<len(cum)-1 and cum[j+1] < s: j+=1
    if j>=len(path_lnglat)-1: j=len(path_lnglat)-2
    a,b=path_lnglat[j], path_lnglat[j+1]
    dseg=(cum[j+1]-cum[j]) or 1.0
    ratio=max(0.0, min(1.0, (s-cum[j])/dseg))
    lng=a[0] + (b[0]-a[0])*ratio
    lat=a[1] + (b[1]-a[1])*ratio
    return lat, lng

def fill_missing_centers_with_fallback(dfg: pd.DataFrame, route_name: str):
    """dfg: 단일 route의 df (lat/lng 결측인 행을 seq 비율 기준 폴백 경로로 보간)"""
    if route_name not in FALLBACK_PATHS: return dfg
    path = FALLBACK_PATHS[route_name]
    if not isinstance(path, list) or len(path)<2: return dfg
    mask = dfg["lat"].isna() | dfg["lng"].isna()
    if not mask.any(): return dfg
    # seq를 이용해 0..1 사이 위치 할당 (동일 간격)
    order = dfg["seq"].rank(method="dense").astype(int)
    n = max(int(order.max()), 1)
    for idx in dfg[mask].index:
        k = int(order.loc[idx]) - 1
        t = 0.0 if n==1 else k / (n-1)
        lat, lng = _interp_on_polyline(path, t)
        if lat is not None and lng is not None:
            dfg.at[idx,"lat"] = lat
            dfg.at[idx,"lng"] = lng
    return dfg

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
                time.sleep(1.0)  # OSM rate-limit 보호
    df["big"]=df["route"].map(ROUTE_TO_BIG).fillna("기타")
    df["big"]=pd.Categorical(df["big"],categories=TOP_ORDER,ordered=True)
    return df

# ─────────────────────────────────────────────────────────────
# 데이터/옵션
# ─────────────────────────────────────────────────────────────
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
st.session_state.setdefault("done_center_ids", set())

BASE_GRAY = [190,190,190]
HIGHLIGHT_COLOR = [230, 57, 70]
ROUTE_COLORS = {r: BASE_GRAY for r in ALL_DEFINED_ROUTES}

# ─────────────────────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────────────────────
tab=st.radio("",["🚴 구간(거리) 추적","📍 인증센터"], horizontal=True, label_visibility="collapsed")

def pick_by_big(all_routes: list[str], key_prefix: str):
    big=st.sidebar.selectbox("대분류", TOP_ORDER, index=0, key=f"{key_prefix}_big")
    if big == "그랜드슬램":
        options = sorted(list(set(all_routes) | set(ALL_DEFINED_ROUTES)))
    else:
        defined=[norm_name(r) for r in BIG_TO_ROUTES.get(big,[])]
        present=[r for r in defined if r in all_routes]
        absent=[r for r in defined if r not in all_routes]
        options=present+[r for r in absent if r in ALL_DEFINED_ROUTES]
    fmt=lambda r: r if r in all_routes else f"{r}  • 데이터없음(폴백)"
    picked=st.sidebar.multiselect("노선(복수 선택 가능)", options, default=options, format_func=fmt, key=f"{key_prefix}_routes")
    return big, [norm_name(r) for r in picked]

# ─────────────────────────────────────────────────────────────
# 1) 구간(거리) 추적
# ─────────────────────────────────────────────────────────────
if tab=="🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")
    all_route_names=sorted(routes["route"].unique().tolist())
    big, picked = pick_by_big(all_route_names, "seg")

    routes2=routes.copy()
    if "path" in routes2.columns:
        m=routes2["path"].notna()
        routes2.loc[m,"path"]=routes2.loc[m,"path"].map(parse_path)

    def centers_path(rname:str):
        if centers is None: return None, np.nan
        g=centers[(centers["route"]==rname)].dropna(subset=["lat","lng"]).sort_values("seq")
        if g.empty: return None, np.nan
        pts=g[["lng","lat"]].to_numpy(float).tolist()
        km=float(g["leg_km"].fillna(0).sum()) if ("leg_km" in g.columns and g["leg_km"].notna().any()) \
            else sum(haversine_km(pts[i][1],pts[i][0],pts[i+1][1],pts[i+1][0]) for i in range(len(pts)-1))
        return pts, km

    summary, base_rows, highlight_rows, view_paths = [], [], [], []

    base_for_done = routes[routes["route"].isin(picked)][["route","section","distance_km","id"]].copy()
    base_for_done["완료"]=base_for_done["id"].isin(st.session_state.done_section_ids)
    done_routes = set(base_for_done.loc[base_for_done["완료"], "route"].astype(str))

    for r in picked:
        fb = FALLBACK_PATHS.get(r)
        sub=routes2[routes2["route"]==r]
        src="fallback" if fb else "없음"
        used_points=len(fb) if fb else 0
        disp_km=float(OFFICIAL_TOTALS.get(r, 0.0))

        path_for_base=None
        if not sub.empty and sub["path"].notna().any():
            p=sub["path"].dropna().iloc[0]
            if p and len(p)>=2:
                path_for_base = p
                src="routes.path"; used_points=len(p)
        else:
            p2,k2 = centers_path(r)
            if p2 and len(p2)>=2:
                path_for_base = p2
                src="centers"; used_points=len(p2); disp_km=float(k2) if not np.isnan(k2) else disp_km
            elif fb and len(fb)>=2:
                path_for_base = fb

        if path_for_base:
            base_rows.append({"route": r, "path": path_for_base, "color": ROUTE_COLORS.get(r, BASE_GRAY), "width": 5})
            view_paths.append(path_for_base)

        if r in done_routes and path_for_base:
            highlight_rows.append({"route": r, "path": path_for_base, "color": HIGHLIGHT_COLOR, "width": 9})

        sub_km=float(sub["distance_km"].fillna(0).sum()) if not sub.empty else 0.0
        if sub_km>0: disp_km=sub_km
        summary.append({"route": r, "경로소스": src, "포인트수": used_points, "표시거리(km)": disp_km})

    with st.expander("선택 노선 총거리 요약", expanded=True):
        df_sum = pd.DataFrame(summary)
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

    base=routes[routes["route"].isin(picked)][["route","section","distance_km","id"]].copy()
    base["완료"]=base["id"].isin(st.session_state.done_section_ids)
    edited=st.data_editor(
        base.drop(columns=["id"]),
        use_container_width=True,
        hide_index=True,
        key="editor_routes",
        column_config={"완료": st.column_config.CheckboxColumn(label="완료", default=False)}
    )
    id_map=dict(zip(base["route"].astype(str)+"@"+base["section"].astype(str), base["id"]))
    new_done=set()
    for _,row in edited.iterrows():
        k=f"{row['route']}@{row['section']}"
        if id_map.get(k) and bool(row["완료"]): new_done.add(id_map[k])
    if new_done != st.session_state.done_section_ids:
        st.session_state.done_section_ids=new_done
        st.experimental_rerun()

    base["완료"]=base["id"].isin(st.session_state.done_section_ids)

    total_km=float(base["distance_km"].fillna(0).sum()) if not base.empty else float(pd.DataFrame(summary)["표시거리(km)"].sum())
    done_km=float(base.loc[base["완료"],"distance_km"].fillna(0).sum())
    if done_km==0 and not base.empty:
        done_km=total_km*float(base["완료"].mean())
    left_km=max(total_km-done_km,0.0)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("대분류", big)

    gj_base = make_geojson_lines(base_rows)
    gj_high = make_geojson_lines(highlight_rows)

    layers=[]
    if gj_base["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_base, pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=5))
    if gj_high["features"]:
        layers.append(pdk.Layer("GeoJsonLayer", gj_high, pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=9))

    centers_for_view=None
    if centers is not None:
        g=centers[centers["route"].isin(picked)].dropna(subset=["lat","lng"]).copy()
        if not g.empty:
            centers_for_view=g.copy()
            g["__color"]=[[210,210,210]]*len(g)
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                g.rename(columns={"lat":"latitude","lng":"longitude"}),
                get_position='[longitude, latitude]',
                get_fill_color="__color",
                get_radius=120,
                pickable=True))

    vlat, vlng, vzoom = view_from(view_paths, centers_for_view, base_zoom=7.0 if len(picked)==1 else 5.8)
    st.pydeck_chart(pdk.Deck(layers=layers,
                             initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
                             tooltip={"text": "{properties.route}"}),
                    use_container_width=True)

# ─────────────────────────────────────────────────────────────
# 2) 인증센터 — 온더플라이 지오코딩 + 폴백 좌표 보간 + 기본 회색/완료 빨강
# ─────────────────────────────────────────────────────────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다."); st.stop()

    st.sidebar.header("인증센터 필터")
    _, picked = pick_by_big(sorted(set(routes["route"])|set(centers["route"])|set(ALL_DEFINED_ROUTES)), "cent")

    dfc=centers[centers["route"].isin(picked)].copy()
    dfc=dfc.sort_values(["route","seq","center"]).reset_index(drop=True)

    # (1) 온더플라이 지오코딩
    need_geo = dfc["address"].notna() & (dfc["lat"].isna() | dfc["lng"].isna())
    for idx, row in dfc[need_geo].iterrows():
        lat, lng = geocode(row["address"])
        if lat is not None and lng is not None:
            dfc.at[idx,"lat"], dfc.at[idx,"lng"] = lat, lng

    # (2) 여전히 비어 있으면 폴백 경로로 seq 비율 보간하여 임시 좌표 생성
    for r, g in dfc.groupby("route"):
        if (g["lat"].isna() | g["lng"].isna()).any():
            dfc.loc[g.index] = fill_missing_centers_with_fallback(g.copy(), r)

    dfc["완료"]=dfc["id"].isin(st.session_state.done_center_ids)

    with st.expander("인증센터 체크(간단 편집)", expanded=True):
        cols=["route","seq","center","address","완료"]
        edited=st.data_editor(
            dfc[cols], use_container_width=True, hide_index=True, key="editor_centers",
            column_config={"완료": st.column_config.CheckboxColumn(label="완료", default=False)}
        )

    # 인덱스(label) 기반 안전 반영
    new_done=set()
    for i,_row in edited.iterrows():
        cid=dfc.loc[i,"id"]
        if bool(_row["완료"]): new_done.add(cid)
    if new_done != st.session_state.done_center_ids:
        st.session_state.done_center_ids=new_done
        st.experimental_rerun()

    dfc["완료"]=dfc["id"].isin(st.session_state.done_center_ids)

    # 센터 → 구간(세그먼트)
    seg=[]
    for r,g in dfc.groupby("route"):
        g=g.sort_values("seq").dropna(subset=["lat","lng"])
        rec=g.to_dict("records")
        for i in range(len(rec)-1):
            a,b=rec[i],rec[i+1]
            dist = (float(a.get("leg_km")) if ("leg_km" in g.columns and not pd.isna(a.get("leg_km")))
                    else (haversine_km(a.get("lat"),a.get("lng"),b.get("lat"),b.get("lng")) or 0.0))
            seg.append({
                "route":r,
                "start_center":a["center"],"end_center":b["center"],
                "path":[[float(a.get("lng")), float(a.get("lat"))], [float(b.get("lng")), float(b.get("lat"))]],
                "distance_km":0.0 if pd.isna(dist) else float(dist),
                "done":bool(a["완료"] and b["완료"]),
            })
    seg_df=pd.DataFrame(seg)

    total=float(seg_df["distance_km"].sum()) if not seg_df.empty else 0.0
    done=float(seg_df.loc[seg_df["done"],"distance_km"].sum()) if not seg_df.empty else 0.0
    left=max(total-done,0.0)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("선택 인증센터 수", f"{dfc.shape[0]:,}")
    c2.metric("완료한 인증센터", f"{int(dfc['완료'].sum()):,}")
    c3.metric("센터 기준 누적거리", f"{done:,.1f} km")
    c4.metric("센터 기준 남은 거리", f"{left:,.1f} km")

    # 지도 레이어: (a) 노선 폴백 라인(밑그림), (b) 세그먼트 전체 회색, (c) 완료 세그먼트 빨강, (d) 센터 점(회색/빨강)
    layers=[]

    # (a) 밑그림: 선택 노선 폴백 라인(있으면)
    fb_rows=[{"route": r, "path": FALLBACK_PATHS[r], "color": [80,80,80], "width": 3}
             for r in picked if r in FALLBACK_PATHS and len(FALLBACK_PATHS[r])>=2]
    if fb_rows:
        gj_fb=make_geojson_lines(fb_rows)
        if gj_fb["features"]:
            layers.append(pdk.Layer("GeoJsonLayer", gj_fb, pickable=False,
                                    get_line_color="properties.color",
                                    get_line_width="properties.width",
                                    line_width_min_pixels=3))

    # (b)(c) 세그먼트
    if not seg_df.empty:
        gj_all=make_geojson_lines([{"route": r["route"], "path": r["path"], "color": BASE_GRAY, "width": 4}
                                   for _, r in seg_df.iterrows()])
        if gj_all["features"]:
            layers.append(pdk.Layer("GeoJsonLayer", gj_all, pickable=True,
                                    get_line_color="properties.color",
                                    get_line_width="properties.width",
                                    line_width_min_pixels=4))
        gj_done=make_geojson_lines([{"route": r["route"], "path": r["path"], "color": HIGHLIGHT_COLOR, "width": 6}
                                    for _, r in seg_df[seg_df["done"]].iterrows()])
        if gj_done["features"]:
            layers.append(pdk.Layer("GeoJsonLayer", gj_done, pickable=True,
                                    get_line_color="properties.color",
                                    get_line_width="properties.width",
                                    line_width_min_pixels=6))

    # (d) 센터 점
    geo=dfc.dropna(subset=["lat","lng"]).copy()
    if not geo.empty:
        geo["__color"]=[BASE_GRAY]*len(geo)
        layers.append(pdk.Layer("ScatterplotLayer",
                                geo.rename(columns={"lat":"latitude","lng":"longitude"}),
                                get_position='[longitude, latitude]',
                                get_fill_color="__color",
                                get_radius=140, pickable=True))
        geo_done=geo[geo["완료"]].copy()
        if not geo_done.empty:
            geo_done["__color"]=[HIGHLIGHT_COLOR]*len(geo_done)
            layers.append(pdk.Layer("ScatterplotLayer",
                                    geo_done.rename(columns={"lat":"latitude","lng":"longitude"}),
                                    get_position='[longitude, latitude]',
                                    get_fill_color="__color",
                                    get_radius=220, pickable=True))

    vlat, vlng, vzoom = view_from([], geo, 7.0)
    st.pydeck_chart(pdk.Deck(layers=layers,
                             initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
                             tooltip={"text":"{properties.route}"}),
                    use_container_width=True)
