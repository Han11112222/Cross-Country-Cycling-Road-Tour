# app.py — 국토종주/4대강/그랜드슬램 대-중-소 선택 + 인증센터 체크/지도/거리
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# 0) 공식 총거리(자전거행복나눔 공인거리 기준; 부족 시 centers 파생거리로 대체)
# ──────────────────────────────────────────────────────────────────────────────
OFFICIAL_TOTALS = {
    # 국토종주(주요 구성 노선)
    "아라자전거길": 21,
    "한강종주자전거길(서울구간)": 40,
    "남한강자전거길": 132,
    "새재자전거길": 100,
    "낙동강자전거길": 389,
    # 4대강
    "금강자전거길": 146,
    "영산강자전거길": 133,
    # 그랜드슬램 기타
    "북한강자전거길": 70,
    "섬진강자전거길": 148,
    "오천자전거길": 105,
    "동해안자전거길(강원구간)": 242,
    "동해안자전거길(경북구간)": 76,
    "제주환상": 234,
}

# ──────────────────────────────────────────────────────────────────────────────
# 1) 대-중 분류(요약 표 반영)
#    - 대분류: 국토종주 / 4대강 종주 / 그랜드슬램
#    - 중분류: 실제 노선명(centers.csv / routes.csv 의 route 값과 일치해야 함)
# ──────────────────────────────────────────────────────────────────────────────
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
        "제주환상",
    ],
}
ALL_ROUTES_IN_APP = sorted({r for rr in BIG_TO_ROUTES.values() for r in rr} |
                           set([*OFFICIAL_TOTALS.keys()]))

# 각 route → 대분류(우선순위: 국토종주 > 4대강 > 그랜드슬램)
ROUTE_TO_BIG = {}
for big in ["국토종주", "4대강 종주", "그랜드슬램"]:
    for r in BIG_TO_ROUTES.get(big, []):
        ROUTE_TO_BIG[r] = big

TOP_ORDER = ["국토종주", "4대강 종주", "그랜드슬램", "기타"]

# ──────────────────────────────────────────────────────────────────────────────
# 2) 경로 폴백(좌표가 비어도 대략 그려주기) — [lng, lat]
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# 3) 유틸
# ──────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def parse_path(s):
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 4) CSV 로드
#    routes.csv (필수컬럼: route, section, distance_km)
#    centers.csv (필수컬럼: route, center, address, lat, lng, id, seq)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_routes(src: str | Path | bytes) -> pd.DataFrame:
    df = pd.read_csv(src)
    need = {"route", "section", "distance_km"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"routes.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    for c in ["route", "section", "start", "end"]:
        if c in df.columns: df[c] = df[c].astype(str).str.strip()
    for c in ["distance_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    if "id" not in df.columns:
        df["id"] = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)

    # 대분류 부여
    df["big"] = df["route"].map(ROUTE_TO_BIG).fillna("기타")
    df["big"] = pd.Categorical(df["big"], categories=TOP_ORDER, ordered=True)
    return df

@st.cache_data
def load_centers(src: str | Path | bytes) -> pd.DataFrame | None:
    if src is None:
        return None
    df = pd.read_csv(src)
    need = {"route", "center", "address", "lat", "lng", "id", "seq"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"centers.csv에 다음 컬럼이 필요합니다: {sorted(miss)}")

    for c in ["route", "center", "address", "id"]:
        df[c] = df[c].astype(str).str.strip()
    for c in ["lat", "lng", "seq", "leg_km"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    df["big"] = df["route"].map(ROUTE_TO_BIG).fillna("기타")
    df["big"] = pd.Categorical(df["big"], categories=TOP_ORDER, ordered=True)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 5) 데이터 소스 선택
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("데이터")
use_repo = st.sidebar.radio("불러오기 방식", ["Repo 내 파일", "CSV 업로드"], index=0)
if use_repo == "Repo 내 파일":
    routes_csv = Path("data/routes.csv")
    centers_csv = Path("data/centers.csv")
    if not routes_csv.exists():
        st.error("Repo에 data/routes.csv 가 없습니다. 먼저 CSV를 추가해주세요.")
        st.stop()
    routes = load_routes(routes_csv)
    centers = load_centers(centers_csv) if centers_csv.exists() else None
else:
    up_r = st.sidebar.file_uploader("routes.csv 업로드", type=["csv"], key="routes_up")
    up_c = st.sidebar.file_uploader("centers.csv 업로드(선택)", type=["csv"], key="centers_up")
    if up_r is None:
        st.info("routes.csv를 올리면 시작합니다.")
        st.stop()
    routes = load_routes(up_r)
    centers = load_centers(up_c) if up_c else None

# 세션 스토리지
if "visited_center_ids" not in st.session_state:
    st.session_state.visited_center_ids = set()
if "done_section_ids" not in st.session_state:
    st.session_state.done_section_ids = set()

# ──────────────────────────────────────────────────────────────────────────────
# 6) 탭
# ──────────────────────────────────────────────────────────────────────────────
tab = st.radio("", ["🚴 구간(거리) 추적", "📍 인증센터"], horizontal=True, label_visibility="collapsed")

# 공통: 대-중 선택 위젯
def big_mid_select(source_routes: list[str]) -> tuple[str, str]:
    bigs = ["국토종주", "4대강 종주", "그랜드슬램"]
    big = st.sidebar.selectbox("대분류", bigs, index=0)
    mids_all = [r for r in BIG_TO_ROUTES.get(big, []) if r in source_routes]
    mid = st.sidebar.selectbox("중분류(노선)", options=mids_all, index=0, help="노선을 선택하면 소분류(인증센터)가 아래에 나와.")
    return big, mid

# ──────────────────────────────────────────────────────────────────────────────
# 7) 구간(거리) 추적(노선 중심)
# ──────────────────────────────────────────────────────────────────────────────
if tab == "🚴 구간(거리) 추적":
    st.sidebar.header("구간 선택")

    # 대-중(노선) 선택
    all_route_names = sorted(routes["route"].unique().tolist())
    big, picked_route = big_mid_select(all_route_names)

    # 이 노선의 구간 목록
    df = routes[routes["route"] == picked_route].copy()
    if df.empty:
        st.warning("선택한 노선에 routes 데이터가 없습니다.")
        st.stop()

    # 파서: routes.path → __path
    df["__path"] = None
    if "path" in df.columns:
        try:
            df.loc[df["path"].notna(), "__path"] = df.loc[df["path"].notna(), "path"].map(parse_path)
        except Exception:
            pass

    # 인증센터 기반 파생 경로/거리
    def centers_polyline_and_km(route_name: str):
        if centers is None:
            return None, np.nan
        g = centers[(centers["route"] == route_name)].dropna(subset=["lat", "lng"]).sort_values("seq")
        if g.empty:
            return None, np.nan
        pts = g[["lng", "lat"]].to_numpy(dtype=float)
        path = pts.tolist()
        if "leg_km" in g.columns and g["leg_km"].notna().any():
            km = float(g["leg_km"].fillna(0).sum())
        else:
            km = 0.0
            for i in range(len(pts) - 1):
                km += haversine_km(pts[i][1], pts[i][0], pts[i + 1][1], pts[i + 1][0])
        return path, km

    p_centers, km_centers = centers_polyline_and_km(picked_route)

    # 표시용 거리(우선순위: routes.distance_km 합계 > OFFICIAL_TOTALS > centers 파생거리)
    km_routes = float(df["distance_km"].fillna(0).sum())
    display_km = km_routes if km_routes > 0 else float(OFFICIAL_TOTALS.get(picked_route, km_centers if not np.isnan(km_centers) else 0.0))

    # 요약
    with st.expander("선택 노선 요약", expanded=True):
        st.dataframe(pd.DataFrame({"route": [picked_route], "표시 거리(km)": [display_km]}),
                     use_container_width=True, hide_index=True)

    # 완료 체크(섹션 단위)
    df["완료"] = df["id"].isin(st.session_state.done_section_ids)
    edited = st.data_editor(df[["route", "section", "distance_km", "완료"]],
                            use_container_width=True, hide_index=True, key="editor_sections")
    # 반영
    new_done = set()
    for _, r in edited.iterrows():
        key = (str(picked_route) + "@" + str(r["section"])).replace(" ", "")
        real_id = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)
        id_map = dict(zip(real_id, df["id"]))
        if bool(r["완료"]) and key in id_map:
            new_done.add(id_map[key])
    st.session_state.done_section_ids = new_done
    df["완료"] = df["id"].isin(st.session_state.done_section_ids)

    # KPI
    total_km = float(df["distance_km"].fillna(0).sum())
    if total_km == 0:  # 섹션별 km가 없는 경우 표시용 거리 사용
        total_km = display_km
    done_km = float(df[df["완료"]]["distance_km"].fillna(0).sum())
    if done_km == 0 and not df[df["완료"]].empty and total_km == display_km:
        # 섹션별 km가 없고 표시용으로만 운영하는 경우, 완료비율로 근사 (완료/전체 섹션 비율)
        done_ratio = df["완료"].mean()
        done_km = display_km * float(done_ratio)
    left_km = max(total_km - done_km, 0.0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 노선 총거리(표시)", f"{display_km:,.1f} km")
    c2.metric("완료 누적거리", f"{done_km:,.1f} km")
    c3.metric("남은 거리", f"{left_km:,.1f} km")
    c4.metric("대분류", big)

    # 지도(노선 한 번만 그리기)
    path_to_draw = None
    used_fallback = False
    if df["__path"].notna().any():
        path_to_draw = df["__path"].dropna().iloc[0]
    if path_to_draw is None and p_centers is not None:
        path_to_draw = p_centers
    if path_to_draw is None:
        path_to_draw = FALLBACK_PATHS.get(picked_route)
        used_fallback = path_to_draw is not None

    layers = []
    if path_to_draw:
        paths_df = pd.DataFrame([{"route": picked_route, "path": path_to_draw, "__color": [28, 200, 138]}])
        layers.append(pdk.Layer("PathLayer", paths_df, get_path="path", get_color="__color",
                                width_scale=3, width_min_pixels=3, pickable=True))

    # 인증센터 마커(회색)
    if centers is not None:
        g = centers[(centers["route"] == picked_route)].dropna(subset=["lat", "lng"]).copy()
        if not g.empty:
            g["__color"] = [[200, 200, 200]] * len(g)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    g.rename(columns={"lat": "latitude", "lng": "longitude"}),
                    get_position='[longitude, latitude]',
                    get_fill_color="__color",
                    get_radius=120,
                    pickable=True,
                )
            )

    # 뷰포인트 — ★ KeyError 방지: lat/lng 로 계산
    if centers is not None:
        geo = centers[(centers["route"] == picked_route)].dropna(subset=["lat", "lng"])
        if not geo.empty:
            vlat, vlng = float(geo["lat"].mean()), float(geo["lng"].mean())
        else:
            vlat, vlng = 36.2, 127.5
    else:
        vlat, vlng = 36.2, 127.5

    st.pydeck_chart(
        pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=7),
                 tooltip={"text": "{route}"}),
        use_container_width=True,
    )
    if used_fallback:
        st.caption("ℹ️ 좌표가 없어 임시(폴백) 경로로 표시함.")

# ──────────────────────────────────────────────────────────────────────────────
# 8) 인증센터(소분류) — 체크 시 지도/거리 반영
# ──────────────────────────────────────────────────────────────────────────────
else:
    if centers is None:
        st.info("data/centers.csv 를 추가하면 인증센터 탭이 활성화됩니다.")
        st.stop()

    st.sidebar.header("인증센터 선택")

    # 대-중(노선) 선택
    centers_routes = sorted(centers["route"].dropna().unique().tolist())
    big, picked_route = big_mid_select(centers_routes)

    # 선택 노선의 인증센터 테이블
    dfc = centers[centers["route"] == picked_route].copy()
    dfc = dfc.sort_values(["seq", "center"], na_position="last").reset_index(drop=True)

    # 방문 체크 동기화
    dfc["완료"] = dfc["id"].isin(st.session_state.visited_center_ids)

    with st.expander("인증센터 체크(소분류)", expanded=True):
        view_cols = ["seq", "center", "address", "완료"]
        edited = st.data_editor(dfc[view_cols], use_container_width=True, hide_index=True, key="editor_centers_small")

    # 반영
    new_set = set()
    for i, row in edited.iterrows():
        cid = dfc.iloc[i]["id"]
        if bool(row["완료"]):
            new_set.add(cid)
    st.session_state.visited_center_ids = new_set
    dfc["완료"] = dfc["id"].isin(st.session_state.visited_center_ids)

    # 세그먼트(센터 i → i+1) 구성 + 완료 구간 계산(양끝 모두 체크된 경우 완료)
    seg_rows = []
    recs = dfc.sort_values("seq").to_dict("records")
    for i in range(len(recs) - 1):
        a, b = recs[i], recs[i + 1]
        if not pd.isna(a.get("leg_km")):
            dist = float(a["leg_km"])
        else:
            dist = haversine_km(a.get("lat"), a.get("lng"), b.get("lat"), b.get("lng"))
            if pd.isna(dist): dist = 0.0
        seg_rows.append({
            "route": picked_route,
            "start_center": a["center"], "end_center": b["center"],
            "start_lat": a.get("lat"), "start_lng": a.get("lng"),
            "end_lat": b.get("lat"), "end_lng": b.get("lng"),
            "distance_km": dist,
            "done": bool(a["완료"] and b["완료"]),
        })
    seg_df = pd.DataFrame(seg_rows)

    # KPI(전체 라이딩 거리 = 완료 세그먼트 합)
    if not seg_df.empty:
        total_km_centers = float(seg_df["distance_km"].sum())  # 노선 기준 총 세그먼트 길이
        done_km_centers = float(seg_df.loc[seg_df["done"], "distance_km"].sum())
        left_km_centers = max(total_km_centers - done_km_centers, 0.0)
    else:
        total_km_centers = done_km_centers = left_km_centers = 0.0

    # OFFICIAL_TOTALS 비교값
    official_km = float(OFFICIAL_TOTALS.get(picked_route, total_km_centers))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선택 인증센터 수", f"{dfc.shape[0]:,}")
    c2.metric("완료한 인증센터", f"{int(dfc[dfc['완료']].shape[0]):,}")
    c3.metric("센터 기준 누적거리(완료)", f"{done_km_centers:,.1f} km")
    c4.metric("공식 총거리(참고)", f"{official_km:,.1f} km")

    # 지도: 완료 세그먼트(초록) / 미완료(빨강) + 센터 마커(완료:초록, 미완료:빨강)
    layers = []
    if not seg_df.empty:
        for flag, color in [(True, [28, 200, 138]), (False, [230, 57, 70])]:
            src = seg_df[seg_df["done"] == flag].copy()
            if src.empty: continue
            src["__path"] = src.apply(lambda r: [[r["start_lng"], r["start_lat"]], [r["end_lng"], r["end_lat"]]], axis=1)
            src["__color"] = [color] * len(src)
            layers.append(pdk.Layer("PathLayer", src, get_path="__path", get_color="__color",
                                    width_scale=3, width_min_pixels=3, pickable=True))

    geo = dfc.dropna(subset=["lat", "lng"]).copy()
    if not geo.empty:
        geo["__color"] = geo["완료"].map(lambda b: [28, 200, 138] if b else [230, 57, 70])
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                geo.rename(columns={"lat": "latitude", "lng": "longitude"}),
                get_position='[longitude, latitude]',
                get_fill_color="__color",
                get_radius=160,
                pickable=True,
            )
        )
        vlat, vlng = float(geo["lat"].mean()), float(geo["lng"].mean())  # ★ latitude/longitude KeyError 방지
    else:
        # 센터 좌표가 전무하면 폴백 경로 평균 또는 전국 중심
        fallback = FALLBACK_PATHS.get(picked_route)
        if fallback:
            arr = np.array(fallback, dtype=float)
            vlng, vlat = float(arr[:,0].mean()), float(arr[:,1].mean())
        else:
            vlat, vlng = 36.2, 127.5

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=vlat, longitude=vlng, zoom=7),
            tooltip={"text": "{route}\n{start_center} → {end_center}"},
        ),
        use_container_width=True,
    )
