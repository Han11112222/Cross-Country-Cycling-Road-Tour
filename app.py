# app.py — Country Cycling Route Tracker (국토종주 누적거리 트래커)
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# -----------------------------
# 데이터 불러오기 (Repo / 업로드 겸용)
# -----------------------------
@st.cache_data
def load_routes(src: str | Path | bytes) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)
    # 보정: id 없으면 route+section으로 자동 생성
    if "id" not in df.columns:
        df["id"] = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)
    # 숫자/좌표형 캐스팅
    for c in ["distance_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

st.sidebar.header("데이터")
use_repo = st.sidebar.radio("불러오기 방식", ["Repo 내 파일", "CSV 업로드"], index=0)
if use_repo == "Repo 내 파일":
    default_csv = "data/routes.csv"
    if not Path(default_csv).exists():
        st.warning("Repo에 data/routes.csv 가 없어요. 우선 샘플을 쓰고 시작합니다.")
        csv_bytes = (Path(__file__).parent / "data" / "routes.sample.csv").read_bytes() if (Path(__file__).parent / "data" / "routes.sample.csv").exists() else None
        if csv_bytes is None:
            st.stop()
        routes = load_routes(csv_bytes)
    else:
        routes = load_routes(default_csv)
else:
    up = st.sidebar.file_uploader("routes.csv 업로드", type=["csv"])
    if up is None:
        st.info("CSV를 올리면 시작합니다.")
        st.stop()
    routes = load_routes(up)

# -----------------------------
# 진행상태 저장/불러오기
# -----------------------------
if "done_ids" not in st.session_state:
    st.session_state.done_ids: set[str] = set()

with st.sidebar.expander("진행상태 저장/불러오기", expanded=False):
    up_state = st.file_uploader("진행상태 불러오기(.json)", type=["json"], key="state_up")
    if up_state:
        try:
            st.session_state.done_ids = set(json.load(up_state))
            st.success("불러왔습니다!")
        except Exception as e:
            st.error(f"불러오기 실패: {e}")

    state_json = json.dumps(sorted(list(st.session_state.done_ids)), ensure_ascii=False)
    st.download_button("진행상태 저장(.json)", data=state_json, file_name="progress.json", mime="application/json")

# -----------------------------
# 사이드바 필터
# -----------------------------
st.sidebar.header("구간 선택")
categories = ["전체구간", "4대강", "제주도환상길", "동해안 자전거길", "기타"]
if "category" not in routes.columns:
    routes["category"] = "전체구간"

cat = st.sidebar.selectbox("대분류", options=sorted(routes["category"].unique().tolist() + list(set(categories) - set(routes["category"]))))

df = routes.copy()
if cat != "전체구간":
    df = df[df["category"] == cat]

route_names = sorted(df["route"].unique().tolist())
route_pick = st.sidebar.multiselect("노선(복수 선택 가능)", options=route_names, default=route_names)

df = df[df["route"].isin(route_pick)].copy()

# -----------------------------
# 완료 체크 UI (테이블)
# -----------------------------
df["완료"] = df["id"].isin(st.session_state.done_ids)
# 편집 후 반영될 수 있도록 키 필요
edited = st.data_editor(
    df[["category", "route", "section", "distance_km", "완료"]],
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="editor",
)

# 데이터 에디터의 완료 체크 결과 → 상태 반영
# (표에서 줄이 추가/삭제될 수 있으므로 id 매칭은 route+section으로 복원)
merge_key = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)
id_map = dict(zip(merge_key, df["id"]))
new_done_ids = set()
for _, row in edited.iterrows():
    k = str(row["route"]) + "@" + str(row["section"])
    k = k.replace(" ", "")
    _id = id_map.get(k)
    if _id and bool(row["완료"]):
        new_done_ids.add(_id)
st.session_state.done_ids = new_done_ids

# -----------------------------
# KPI 카드
# -----------------------------
total_km = float(df["distance_km"].sum())
done_km = float(df[df["id"].isin(st.session_state.done_ids)]["distance_km"].sum())
left_km = max(total_km - done_km, 0.0)
col1, col2, col3, col4 = st.columns(4)
col1.metric("선택 구간 총거리", f"{total_km:,.1f} km")
col2.metric("완료 누적거리", f"{done_km:,.1f} km")
col3.metric("남은 거리", f"{left_km:,.1f} km")
pct = (done_km / total_km * 100) if total_km > 0 else 0
col4.metric("완료율", f"{pct:,.1f}%")

# -----------------------------
# 지도 그리기 (pydeck)
# - path 컬럼이 있으면 경로(PathLayer)
# - 없으면 시작/끝점 마커(ScatterplotLayer)
# -----------------------------
def to_path(row):
    # path: '[ [lng,lat], [lng,lat], ... ]' 형식 문자열이면 파싱
    if "path" in row and pd.notna(row["path"]):
        try:
            val = json.loads(row["path"])
            if isinstance(val, list):
                return val
        except Exception:
            pass
    # fallback: 시작~끝 직선
    if pd.notna(row["start_lng"]) and pd.notna(row["start_lat"]) and pd.notna(row["end_lng"]) and pd.notna(row["end_lat"]):
        return [[row["start_lng"], row["start_lat"]], [row["end_lng"], row["end_lat"]]]
    return None

df["__path"] = df.apply(to_path, axis=1)
paths = df[df["__path"].notna()].copy()

# 중심점 계산(맵 초기 시야)
def mid_lon_lat(row):
    xs = [row.get("start_lng"), row.get("end_lng")]
    ys = [row.get("start_lat"), row.get("end_lat")]
    xs = [x for x in xs if pd.notna(x)]
    ys = [y for y in ys if pd.notna(y)]
    if xs and ys:
        return np.mean(xs), np.mean(ys)
    return None

centers = [mid_lon_lat(r) for _, r in df.iterrows()]
centers = [c for c in centers if c]
center_lng, center_lat = (127.5, 36.2) if not centers else (float(np.mean([c[0] for c in centers])), float(np.mean([c[1] for c in centers])))

layers = []

if not paths.empty:
    # 완료 여부에 따라 색상
    paths["color"] = np.where(paths["id"].isin(st.session_state.done_ids), [28, 200, 138], [230, 57, 70])
    layers.append(
        pdk.Layer(
            "PathLayer",
            paths,
            get_path="__path",
            get_color="color",
            width_scale=3,
            width_min_pixels=3,
            pickable=True,
        )
    )
else:
    # 마커 레이어 (시작/끝)
    pts = []
    for _, r in df.iterrows():
        for (lng, lat, label) in [
            (r.get("start_lng"), r.get("start_lat"), "start"),
            (r.get("end_lng"), r.get("end_lat"), "end"),
        ]:
            if pd.notna(lng) and pd.notna(lat):
                pts.append({
                    "lng": float(lng), "lat": float(lat),
                    "name": f"{r['route']} / {r['section']} ({label})",
                    "done": bool(r["id"] in st.session_state.done_ids)
                })
    if pts:
        pts_df = pd.DataFrame(pts)
        pts_df["color"] = np.where(pts_df["done"], [28,200,138], [230,57,70])
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                pts_df,
                get_position='[lng, lat]',
                get_fill_color='color',
                get_radius=150,
                pickable=True,
            )
        )

view = pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7)
r = pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{name}"})
st.pydeck_chart(r, use_container_width=True)

st.caption("💡 *경로를 지도에 그리려면 CSV에 path 열을 넣어 좌표 배열(JSON)로 제공하세요. 없으면 시작/끝 좌표를 점으로 표시합니다.*")
