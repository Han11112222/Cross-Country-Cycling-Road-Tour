# app.py — v15: 상태 초기화 버튼, 완료 빠른 토글, 회색 베이스 + 강조색 완료,
#               경로 소스 우선순위 선택, 안전 뷰계산 고정

from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import requests

BUILD_TAG = "2025-08-31-v15"

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")
st.caption(f"BUILD: {BUILD_TAG}")

# ───────────────── 기본 데이터 ─────────────────
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
def norm_name(s: str) -> str:
    s = str(s).strip()
    return "제주환상" if s == "제주환상자전거길" else s
ROUTE_TO_BIG = {norm_name(r): big for big, rs in BIG_TO_ROUTES.items() for r in rs}
ALL_DEFINED_ROUTES = sorted({norm_name(r) for v in BIG_TO_ROUTES.values() for r in v})

# 폴백 좌표(대략 경로)
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

# ───────────────── 유틸 ─────────────────
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
    except Exception: pass
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
        except Exception: pass
    if not pts: return 36.2, 127.5, base_zoom
    arr=np.asarray(pts,float).reshape(-1,2)
    vlat, vlng = float(np.mean(arr[:,0])), float(np.mean(arr[:,1]))
    span = 0.0 if arr.shape[0]<2 else max(arr[:,0].ptp(), arr[:,1].ptp())
    zoom = 6.0 if span>3 else base_zoom
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

def items_to_path_df(items):
    if not items: return pd.DataFrame(columns=["route","path","color","width"])
    df=pd.DataFrame(items)
    for c in ["route","path","color","width"]:
        if c not in df.columns: df[c]=None
    return df[["route","path","color","width"]]

def items_to_points_df(items):
    rows=[]
    for it in (items or []):
        route=it.get("route","")
        color=it.get("color",[180,180,180])
        for lng,lat in (it.get("path") or []):
            try:
                lng=float(lng); lat=float(lat)
            except Exception:
                continue
            if not (pd.isna(lng) or pd.isna(lat)):
                rows.append({"route":route,"longitude":lng,"latitude":lat,"color":color})
    return pd.DataFrame(rows)

# ───────────────── 데이터 로딩 ─────────────────
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

# ───────────────── 사이드바 ─────────────────
st.sidebar.header("데이터")
use_repo=st.sidebar.radio("불러오기 방식",["Repo 내 파일","CSV 업로드"],index=0)
auto_geo=st.sidebar.toggle("주소 → 좌표 자동보정(지오코딩)", value=True)
if st.sidebar.button("↺ 상태 초기화(체크/편집키 포함)", use_container_width=True):
    for k in list(st.session_state.keys()):
        if k.startswith("editor_") or k in ("done_section_ids","done_center_ids","quick_done"):
            st.session_state.pop(k, None)
    st.cache_data.clear()
    st.rerun()

show_debug=st.sidebar.checkbox("디버그 보기", value=False)

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

# 세션 기본값
st.session_state.setdefault("done_section_ids", set())
st.session_state.setdefault("done_center_ids", set())

ROUTE_COLORS = {
    "아라자전거길": [150,150,150],  # 기본 회색표현은 따로, 강조는 아래에서 지정
    "한강종주자전거길(서울구간)": [150,150,150],
    "남한강자전거길": [150,150,150],
    "새재자전거길": [150,150,150],
    "낙동강자전거길": [150,150,150],
    "금강자전거길": [150,150,150],
    "영산강자전거길": [150,150,150],
    "북한강자전거길": [150,150,150],
    "섬진강자전거길": [150,150,150],
    "오천자전거길": [150,150,150],
    "동해안자전거길(강원구간)": [150,150,150],
    "동해안자전거길(경북구간)": [150,150,150],
    "제주환상": [150,150,150],
}
HIGHLIGHT = [255, 80, 80]   # 완료 강조색(빨강)
BASE_GREY  = [180, 180, 180]  # 전체회색

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

# ───────── 1) 구간(거리) 추적 ─────────
if tab=="🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")
    all_route_names=sorted(routes["route"].unique().tolist())
    big, picked = pick_by_big(all_route_names + ALL_DEFINED_ROUTES, "seg", use_defined=True)

    # 경로 소스 우선순위
    order = st.sidebar.selectbox(
        "경로 소스 우선순위",
        ["routes > centers > fallback", "centers > routes > fallback"],
        index=0
    )

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

    def choose_path(r):
        """우선순위대로 path 결정"""
        sub=routes2[routes2["route"]==r]
        p_routes = sub["path"].dropna().iloc[0] if (not sub.empty and sub["path"].notna().any()) else None
        p_centers, km_centers = centers_path(r)
        fb = FALLBACK_PATHS.get(r)

        chain = ["routes","centers","fallback"] if order.startswith("routes") \
                else ["centers","routes","fallback"]

        for k in chain:
            if k=="routes" and p_routes and len(p_routes)>=2:
                return p_routes, "routes"
            if k=="centers" and p_centers and len(p_centers)>=2:
                return p_centers, "centers"
            if k=="fallback" and fb and len(fb)>=2:
                return fb, "fallback"
        return None, "none"

    # 표/완료 관리
    base=routes[routes["route"].isin(picked)][["route","section","distance_km","id"]].copy()
    base["완료"]=base["id"].isin(st.session_state.done_section_ids)

    # 빠른 완료 토글(멀티)
    quick_done = st.sidebar.multiselect(
        "완료 처리(빠른 선택)",
        options=base["route"].tolist(),
        default=[r for r,done in zip(base["route"], base["완료"]) if done],
        key="quick_done"
    )
    quick_done_ids = set(base.loc[base["route"].isin(quick_done),"id"].tolist())

    # 표(완료 체크는 여기서도 가능)
    edited = st.data_editor(
        base.drop(columns=["id"]),
        key="editor_routes",
        hide_index=True,
        use_container_width=True,
        column_config={
            "route":"route",
            "section":"section",
            "distance_km":"distance_km",
            "완료": st.column_config.CheckboxColumn("완료"),
        }
    )

    # 최종 완료집합 = 표 체크 ∪ 빠른 토글
    id_map=dict(zip(base["route"].astype(str)+"@"+base["section"].astype(str), base["id"]))
    done_from_table=set()
    for _,row in edited.iterrows():
        if bool(row["완료"]):
            k=f'{row["route"]}@{row["section"]}'
            if k in id_map: done_from_table.add(id_map[k])
    st.session_state.done_section_ids = set(done_from_table) | set(quick_done_ids)
    base["완료"]=base["id"].isin(st.session_state.done_section_ids)

    # 요약 수치
    total_km=float(base["distance_km"].fillna(0).sum()) if not base.empty else float(sum(OFFICIAL_TOTALS.get(r,0) for r in picked))
    done_km=float(base.loc[base["완료"],"distance_km"].fillna(0).sum())
    left_km=max(total_km-done_km,0.0)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("대분류", big)

    # 지도 레이어 조립: 베이스(회색) + 완료(강조)
    base_rows, hl_rows, view_paths = [], [], []
    for r in picked:
        p, src = choose_path(r)
        if p and len(p)>=2:
            view_paths.append(p)
            base_rows.append({"route": r, "path": p, "color": BASE_GREY, "width": 5})
            # 완료면 강조
            rid = (routes[(routes["route"]==r)]["id"]).iloc[0] if (r in routes["route"].values) else None
            if rid in st.session_state.done_section_ids:
                hl_rows.append({"route": r, "path": p, "color": HIGHLIGHT, "width": 7})

    # 레이어 만들기
    layers=[]
    # 베이스(회색)
    if base_rows:
        gj_base = make_geojson_lines(base_rows)
        layers.append(pdk.Layer("GeoJsonLayer", gj_base,
                                pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=4))
    # 완료(강조)
    if hl_rows:
        gj_hl = make_geojson_lines(hl_rows)
        layers.append(pdk.Layer("GeoJsonLayer", gj_hl,
                                pickable=True,
                                get_line_color="properties.color",
                                get_line_width="properties.width",
                                line_width_min_pixels=6))

    # 점(좌표 보강용)
    pts_df = items_to_points_df(base_rows)
    if not pts_df.empty:
        layers.append(pdk.Layer("ScatterplotLayer", pts_df,
                                get_position='[longitude, latitude]',
                                get_fill_color=[220,220,220],
                                get_radius=60,
                                pickable=True))

    # 뷰
    centers_for_view=None
    if centers is not None:
        g=centers[centers["route"].isin(picked)].dropna(subset=["lat","lng"]).copy()
        centers_for_view = None if g.empty else g
    vlat, vlng, vzoom = view_from_safe(view_paths, centers_for_view, base_zoom=7.0 if len(picked)==1 else 5.8)

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text": "{route}"},
    ), use_container_width=True)

# ───────── 2) 인증센터 ─────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다."); st.stop()

    st.sidebar.header("인증센터 필터")
    _, picked = pick_by_big(sorted(set(routes["route"])|set(centers["route"])|set(ALL_DEFINED_ROUTES)), "cent", use_defined=True)

    dfc=centers[centers["route"].isin(picked)].copy()
    dfc=dfc.sort_values(["route","seq","center"]).reset_index(drop=True)
    dfc["완료"]=dfc["id"].isin(st.session_state.done_center_ids)

    with st.expander("인증센터 체크(간단 편집)", expanded=True):
        cols=["route","seq","center","address","완료"]
        edited=st.data_editor(dfc[cols], use_container_width=True, hide_index=True, key="editor_centers")

    new_done=set()
    for i,_row in edited.iterrows():
        cid=dfc.iloc[i]["id"]
        if bool(_row["완료"]): new_done.add(cid)
    st.session_state.done_center_ids=new_done
    dfc["완료"]=dfc["id"].isin(st.session_state.done_center_ids)

    # 구간 라인(센터 간)
    seg=[]
    for r,g in dfc.groupby("route"):
        g=g.sort_values("seq"); rec=g.to_dict("records")
        for i in range(len(rec)-1):
            a,b=rec[i],rec[i+1]
            if pd.isna(a.get("lat")) or pd.isna(a.get("lng")) or pd.isna(b.get("lat")) or pd.isna(b.get("lng")):
                continue
            dist=float(a.get("leg_km")) if not pd.isna(a.get("leg_km")) else (haversine_km(a.get("lat"),a.get("lng"),b.get("lat"),b.get("lng")) or 0.0)
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

    done_items=[{"route": r["route"], "path": r["path"], "color": HIGHLIGHT, "width": 4}
                for _, r in seg_df[seg_df["done"]].iterrows()]
    todo_items=[{"route": r["route"], "path": r["path"], "color": BASE_GREY, "width": 4}
                for _, r in seg_df[~seg_df["done"]].iterrows()]

    layers=[]
    for items in (todo_items, done_items):
        if items:
            layers.append(pdk.Layer("GeoJsonLayer", make_geojson_lines(items),
                                    pickable=True,
                                    get_line_color="properties.color",
                                    get_line_width="properties.width",
                                    line_width_min_pixels=4))

    geo=dfc.dropna(subset=["lat","lng"]).copy()
    if not geo.empty:
        geo["__color"]=geo["완료"].map(lambda b:HIGHLIGHT if b else BASE_GREY)
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            geo.rename(columns={"lat":"latitude","lng":"longitude"}),
            get_position='[longitude, latitude]',
            get_fill_color="__color",
            get_radius=160,
            pickable=True,
        ))

    vlat, vlng, vzoom = view_from_safe([], geo, 7.0)
    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
        tooltip={"text":"{route}\n{properties.start_center} → {properties.end_center}"},
    ), use_container_width=True)
