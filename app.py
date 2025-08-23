# app.py — Country Cycling Route Tracker (국토종주 누적거리 트래커)
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="국토종주 누적거리 트래커", layout="wide")

# -----------------------------
# 1) 데이터 로드
# -----------------------------
@st.cache_data
def load_routes(src: str | Path | bytes) -> pd.DataFrame:
    df = pd.read_csv(src) if isinstance(src, (str, Path)) else pd.read_csv(src)

    # 필수 컬럼 체크
    need = {"category", "route", "section", "distance_km"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"CSV에 다음 컬럼이 필요합니다: {sorted(miss)}")

    # 문자열/숫자 정리
    for c in ["category", "route", "section", "start", "end"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["distance_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # id 없으면 route+section으로 생성
    if "id" not in df.columns:
        df["id"] = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)

    # ⚠️ 자동 분류(4대강/동해/제주/내륙) 제거: CSV 값을 그대로 씁니다
    return df


# -----------------------------
# 2) 데이터 선택(Repo/업로드)
# -----------------------------
st.sidebar.header("데이터")
use_repo = st.sidebar.radio("불러오기 방식", ["Repo 내 파일", "CSV 업로드"], index=0)

if use_repo == "Repo 내 파일":
    default_csv = "data/routes.csv"
    if Path(default_csv).exists():
        routes = load_routes(default_csv)
    else:
        st.error("Repo에 data/routes.csv 가 없습니다. 먼저 CSV를 추가해주세요.")
        st.stop()
else:
    up = st.sidebar.file_uploader("routes.csv 업로드", type=["csv"])
    if up is None:
        st.info("CSV를 올리면 시작합니다.")
        st.stop()
    routes = load_routes(up)


# -----------------------------
# 3) 진행 상태 저장/불러오기
# -----------------------------
if "done_ids" not in st.session_state:
    st.session_state.done_ids = set()

with st.sidebar.expander("진행상태 저장/불러오기", expanded=False):
    up_state = st.file_uploader("진행상태 불러오기(.json)", type=["json"], key="state_up")
    if up_state:
        try:
            st.session_state.done_ids = set(json.load(up_state))
            st.success("진행상태를 불러왔습니다.")
        except Exception as e:
            st.error(f"불러오기 실패: {e}")

    st.download_button(
        "진행상태 저장(.json)",
        data=json.dumps(sorted(list(st.session_state.done_ids)), ensure_ascii=False),
        file_name="progress.json",
        mime="application/json",
    )


# -----------------------------
# 4) 필터(좌측 사이드바)
# -----------------------------
st.sidebar.header("구간 선택")

cat_list = ["전체구간"] + sorted(routes["category"].dropna().unique().tolist())
cat = st.sidebar.selectbox("대분류", options=cat_list, index=0)

df = routes.copy()
if cat != "전체구간":
    df = df[df["category"] == cat]

route_names = sorted(df["route"].dropna().unique().tolist())
if not route_names:
    st.warning("선택한 대분류에 노선이 없습니다.")
    st.stop()

route_pick = st.sidebar.multiselect("노선(복수 선택 가능)", options=route_names, default=route_names)
if len(route_pick) == 0:
    st.info("왼쪽에서 노선을 최소 1개 이상 선택하세요.")
    st.stop()

df = df[df["route"].isin(route_pick)].copy()
st.caption(f"🔎 필터: 카테고리 **{cat}**, 노선 **{', '.join(route_pick)}**")


# -----------------------------
# 5) 완료 체크 UI
# -----------------------------
df["완료"] = df["id"].isin(st.session_state.done_ids)

edited = st.data_editor(
    df[["category", "route", "section", "distance_km", "완료"]],
    use_container_width=True,
    hide_index=True,
    key="editor",
)

# 에디터 결과 → 상태 반영 (route+section 기준 매칭)
merge_key = (df["route"].astype(str) + "@" + df["section"].astype(str)).str.replace(r"\s+", "", regex=True)
id_map = dict(zip(merge_key, df["id"]))
new_done = set()
for _, row in edited.iterrows():
    k = (str(row["route"]) + "@" + str(row["section"])).replace(" ", "")
    _id = id_map.get(k)
    if _id and bool(row["완료"]):
        new_done.add(_id)
st.session_state.done_ids = new_done


# -----------------------------
# 6) KPI
# -----------------------------
total_km = float(df["distance_km"].sum())
done_km  = float(df[df["id"].isin(st.session_state.done_ids)]["distance_km"].sum())
left_km  = max(total_km - done_km, 0.0)
c1, c2, c3, c4 = st.columns(4)
c1.metric("선택 구간 총거리", f"{total_km:,.1f} km")
c2.metric("완료 누적거리", f"{done_km:,.1f} km")
c3.metric("남은 거리", f"{left_km:,.1f} km")
c4.metric("완료율", f"{(done_km/total_km*100 if total_km>0 else 0):.1f}%")


# -----------------------------
# 7) 지도 (pydeck)
# -----------------------------
def parse_path(s):
    """CSV path 컬럼에 JSON 문자열이 있으면 파싱해서 반환"""
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    return None

df["__path"] = None
if "path" in df.columns:
    df["__path"] = df["path"].dropna().map(parse_path)

# path가 있는 것만 선(PathLayer)으로 그림
paths = df[df["__path"].notna()].copy()

# path가 없으면 점(Scatter)으로 표시 (시작/끝 좌표가 있을 때만)
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
pts_df = pd.DataFrame(pts)

# 지도 중심
if len(pts_df) > 0:
    center_lng, center_lat = float(pts_df["lng"].mean()), float(pts_df["lat"].mean())
else:
    center_lng, center_lat = 127.5, 36.2

layers = []
if not paths.empty:
    paths["__color"] = paths["id"].apply(lambda x: [28, 200, 138] if x in st.session_state.done_ids else [230, 57, 70])
    layers.append(
        pdk.Layer(
            "PathLayer",
            paths,
            get_path="__path",
            get_color="__color",
            width_scale=3,
            width_min_pixels=3,
            pickable=True,
        )
    )

if not pts_df.empty:
    pts_df["__color"] = pts_df["done"].map(lambda b: [28,200,138] if b else [230,57,70])
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            pts_df,
            get_position='[lng, lat]',
            get_fill_color='__color',
            get_radius=150,
            pickable=True,
        )
    )

view = pdk.ViewState(latitude=center_lat, longitude=center_lng, zoom=7)
deck = pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{name}"})
st.pydeck_chart(deck, use_container_width=True)

st.caption("💡 선으로 보이게 하려면 CSV의 path 열에 [ [lng,lat], [lng,lat], ... ] 형식 JSON을 넣어주세요. path가 없으면 시작/끝 좌표가 마커로 표시됩니다.")
