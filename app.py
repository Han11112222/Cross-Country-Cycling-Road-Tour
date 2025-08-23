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

    # 문자열 전처리
    for c in ["category", "route", "section", "start", "end"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 숫자 캐스팅
    for c in ["distance_km", "start_lat", "start_lng", "end_lat", "end_lng"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # id 없으면 route+section으로 생성
    if "id" not in df.columns:
        df["id"] = (
            df["route"].astype(str) + "@" + df["section"].astype(str)
        ).str.replace(r"\s+", "", regex=True)

    # ===== 대분류 자동 산정(노선명 기준) =====
    four_rivers = {
        "한강종주자전거길", "한강종주자전거길(서울구간)",
        "금강자전거길", "영산강자전거길", "낙동강자전거길",
    }

    def big_category_by_route(route: str) -> str:
        r = str(route)
        if "동해안" in r:
            return "동해안자전거길"
        if "제주" in r:
            return "제주환상"
        if r in four_rivers:
            return "4대강"
        return "내륙/연결"

    # CSV의 category와 상관없이 통일 규칙 적용
    df["category"] = df["route"].apply(big_category_by_route)

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

CATEGORY_ORDER = ["전체구간", "4대강", "동해안자전거길", "제주환상", "내륙/연결"]
cat = st.sidebar.selectbox("대분류", options=CATEGORY_ORDER, index=0)

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
# 5) 완료 체크 UI (행 단위)
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
# 6) KPI (전체 합계)
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
# 6-1) 노선별 요약(합계/완료/잔여)
# -----------------------------
summary = (
    df.assign(done=df["id"].isin(st.session_state.done_ids).astype(int))
      .groupby("route", as_index=False)
      .agg(total_km=("distance_km","sum"),
           done_km =("distance_km", lambda s: float(s[df.loc[s.index, "id"].isin(st.session_state.done_ids)].sum())))
)
summary["left_km"] = (summary["total_km"] - summary["done_km"]).clip(lower=0)
summary["done_%"]  = np.where(summary["total_km"]>0, summary["done_km"]/summary["total_km"]*100, 0)

st.subheader("노선별 요약")
st.dataframe(
    summary.rename(columns={
        "route": "노선",
        "total_km": "총거리(km)",
        "done_km": "완료(km)",
        "left_km": "잔여(km)",
        "done_%": "완료율(%)"
    }).round({"총거리(km)":1,"완료(km)":1,"잔여(km)":1,"완료율(%)":1}),
    use_container_width=True
)


# -----------------------------
# 7) 지도 (pydeck) — 좌표가 있으면 선/점 표시
# -----------------------------
def to_path(row):
    if "path" in row and pd.notna(row["path"]):
        try:
            val = json.loads(row["path"])
            if isinstance(val, list):
                return val
        except Exception:
            pass
    if (
        pd.notna(row.get("start_lng")) and pd.notna(row.get("start_lat"))
        and pd.notna(row.get("end_lng")) and pd.notna(row.get("end_lat"))
    ):
        return [[row["start_lng"], row["start_lat"]], [row["end_lng"], row["end_lat"]]]
    return None

df["__path"] = df.apply(to_path, axis=1)
paths = df[df["__path"].notna()].copy()

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
center_lng, center_lat = (127.5, 36.2) if not centers else (
    float(np.mean([c[0] for c in centers])),
    float(np.mean([c[1] for c in centers])),
)

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
else:
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

st.caption("💡 표의 거리/구간은 data/routes.csv 내용을 그대로 사용합니다. 필요 시 CSV를 업데이트해 주세요.")
