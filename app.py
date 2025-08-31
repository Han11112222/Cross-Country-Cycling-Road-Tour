# app.py — 리셋 버전 v17r
# - 기본: 회색 전체 노선
# - 체크한 노선만 강조(청록) + 완료거리 합산
# - 1클릭 반영, CSV 없어도 동작(폴백 경로)
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

BUILD_TAG = "2025-09-01-v17r"
st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")
st.caption(f"BUILD: {BUILD_TAG}")

# ───────────────────────── 공인 총거리(표시용) ─────────────────────────
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
ROUTE_TO_BIG = {norm_name(r): big for big, rs in BIG_TO_ROUTES.items() for r in rs}
ALL_DEFINED_ROUTES = sorted({norm_name(r) for v in BIG_TO_ROUTES.values() for r in v})

# ───────────────────────── 폴백 경로([lng,lat]) ─────────────────────────
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

# ───────────────────────── 유틸 ─────────────────────────
def parse_path(s):
    try:
        v=json.loads(s)
        if isinstance(v,list): return v
    except Exception:
        pass
    return None

def view_from(paths, base_zoom=6.0):
    pts=[]
    for p in (paths or []):
        for xy in (p or []):
            try:
                lng, lat = float(xy[0]), float(xy[1])
                if not (np.isnan(lat) or np.isnan(lng)):
                    pts.append([lat, lng])
            except Exception:
                continue
    if not pts:
        return 36.2, 127.5, base_zoom
    arr=np.asarray(pts,float).reshape(-1,2)
    vlat=float(arr[:,0].mean()); vlng=float(arr[:,1].mean())
    span=max(float(arr[:,0].ptp()), float(arr[:,1].ptp()))
    zoom = 6.0 if span>3 else base_zoom
    return vlat, vlng, zoom

# ───────────────────────── 데이터 로딩(선택) ─────────────────────────
@st.cache_data
def load_routes_csv(p:Path):
    df=pd.read_csv(p)
    df["route"]=df["route"].astype(str).str.strip().map(norm_name)
    if "path" in df.columns:
        m=df["path"].notna()
        df.loc[m,"path"]=df.loc[m,"path"].map(parse_path)
    return df

routes_csv = Path("data/routes.csv")
routes_df = load_routes_csv(routes_csv) if routes_csv.exists() else None

# ───────────────────────── 사이드바 ─────────────────────────
st.sidebar.header("데이터")
if st.sidebar.button("↻ 캐시 초기화", use_container_width=True):
    st.cache_data.clear(); st.rerun()

st.sidebar.header("구간 선택")
big = st.sidebar.selectbox("대분류", TOP_ORDER, index=0)
default_names = [norm_name(r) for r in BIG_TO_ROUTES.get(big, [])]
all_names = sorted(set(default_names) | set(ALL_DEFINED_ROUTES))
picked = st.sidebar.multiselect("노선(복수 선택 가능)", all_names, default=default_names)

# ───────────────────────── 테이블 + 체크박스 ─────────────────────────
# 표 행 구성(공식 총거리 우선, 없으면 0)
base_rows = []
for r in picked:
    base_rows.append({
        "route": r,
        "section": "전체구간",
        "distance_km": float(OFFICIAL_TOTALS.get(r, 0.0)),
    })
base = pd.DataFrame(base_rows, columns=["route","section","distance_km"])

# 세션 상태에 완료 체크 저장
st.session_state.setdefault("done_by_route", {})
for r in base["route"]:
    st.session_state["done_by_route"].setdefault(r, False)

# 데이터 에디터(체크 1클릭 반영)
edited = st.data_editor(
    base.assign(완료=[st.session_state["done_by_route"][r] for r in base["route"]]),
    column_config={
        "route": st.column_config.TextColumn("route", disabled=True),
        "section": st.column_config.TextColumn("section", disabled=True),
        "distance_km": st.column_config.NumberColumn("distance_km", disabled=True),
        "완료": st.column_config.CheckboxColumn("완료", default=False),
    },
    use_container_width=True,
    hide_index=True,
    key="editor_routes",
)

# 완료 상태 업데이트
for _, row in edited.iterrows():
    st.session_state["done_by_route"][row["route"]] = bool(row["완료"])

# 거리 지표
total_km = float(edited["distance_km"].fillna(0).sum())
done_mask = edited["route"].map(lambda r: st.session_state["done_by_route"].get(r, False))
done_km = float(edited.loc[done_mask, "distance_km"].fillna(0).sum())
left_km = max(total_km - done_km, 0.0)

c1,c2,c3,c4 = st.columns(4)
c1.metric("선택 구간 총거리(표 합계)", f"{total_km:,.1f} km")
c2.metric("완료 누적거리", f"{done_km:,.1f} km")
c3.metric("남은 거리", f"{left_km:,.1f} km")
c4.metric("대분류", big)

# ───────────────────────── 지도 레이어 구성 ─────────────────────────
# 1) 회색(전체 노선)
grey_items=[]
# 2) 강조(완료 체크 노선)
main_items=[]

for r in picked:
    # 경로 소스 우선순위: routes.csv.path -> 폴백
    path=None
    if routes_df is not None:
        sub=routes_df[routes_df["route"]==r]
        if not sub.empty and ("path" in sub.columns) and sub["path"].notna().any():
            cand = sub["path"].dropna().iloc[0]
            if isinstance(cand, list) and len(cand)>=2:
                path=cand
    if path is None:
        path = FALLBACK_PATHS.get(r)

    # 회색 기본선
    if isinstance(path, list) and len(path)>=2:
        grey_items.append({"route": r, "path": path, "color": [180,180,180, 255], "width": 5})
        if st.session_state["done_by_route"].get(r, False):
            main_items.append({"route": r, "path": path, "color": [28,200,138, 255], "width": 6})

def items_to_df(items):
    if not items: 
        return pd.DataFrame(columns=["route","path","color","width"])
    df=pd.DataFrame(items)
    for c in ["route","path","color","width"]:
        if c not in df.columns: df[c]=None
    return df[["route","path","color","width"]]

grey_df = items_to_df(grey_items)
main_df = items_to_df(main_items)

layers=[]
if not grey_df.empty:
    layers.append(pdk.Layer(
        "PathLayer", grey_df,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=3,
        pickable=True,
    ))
if not main_df.empty:
    layers.append(pdk.Layer(
        "PathLayer", main_df,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=4,
        pickable=True,
    ))

all_paths = [*grey_df["path"].tolist(), *main_df["path"].tolist()]
vlat, vlng, vzoom = view_from(all_paths, base_zoom=6.2)

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=vzoom),
    layers=layers,
    tooltip={"text": "{route}"},
), use_container_width=True)
