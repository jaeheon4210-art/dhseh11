import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="서울 기온 다항회귀 예측기", page_icon="📈", layout="wide")

st.title("📈 1차·3차·9차 다항 회귀 곡선 비교")
st.write("2005년 이전 데이터로만 회귀 곡선을 학습한 후, 미학습 2005년 이후 테스트 데이터에서의 오차(MAE)와 2050년 예상 기온을 측정합니다.")

# 1. 데이터 불러오기 및 전처리
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    df_filtered = df[df["연도"] <= 2025]
    
    yearly_summary = df_filtered.groupby("연도")["평균기온"].agg(
        count="count",
        mean_temp="mean"
    ).reset_index()
    
    # 관측일수 300일 이상 필터링
    valid_data = yearly_summary[yearly_summary["count"] >= 300].copy()
    
    # 고차 곡선 오버플로우 방지를 위한 X 변환 (1908년 기준, 100년 단위 스케일링)
    valid_data["X_scaled"] = (valid_data["연도"] - 1908) / 100.0
    return valid_data

data = load_data()

# 2. 훈련용 / 테스트용 분할 (2005년 기준)
train_df = data[data["연도"] < 2005].copy()
test_df = data[data["연도"] >= 2005].copy()

# 데이터 분할 통계 화면 표시
col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="📚 훈련용 데이터 (Train Set)",
        value=f"{len(train_df)}개 해",
        delta=f"{train_df['연도'].min()}년 ~ {train_df['연도'].max()}년"
    )
with col2:
    st.metric(
        label="🧪 테스트용 데이터 (Test Set)",
        value=f"{len(test_df)}개 해",
        delta=f"{test_df['연도'].min()}년 ~ {test_df['연도'].max()}년"
    )

st.divider()

# 3. 모델 학습 및 평가 (1차, 3차, 9차)
degrees = [1, 3, 9]
eval_results = []
models = {}

X_train = train_df[["X_scaled"]]
y_train = train_df["mean_temp"]

X_test = test_df[["X_scaled"]]
y_test = test_df["mean_temp"]

# 2050년 스케일링 값 계산
x_2050_scaled = pd.DataFrame({"X_scaled": [(2050 - 1908) / 100.0]})

for deg in degrees:
    # 파이프라인으로 다항 피처 생성 및 선형 회귀 결합
    model = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
    model.fit(X_train, y_train)
    
    # 오직 테스트 데이터로만 예측 및 MAE 채점
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    
    # 2050년 기온 예측
    pred_2050 = model.predict(x_2050_scaled)[0]
    
    models[deg] = model
    eval_results.append({
        "곡선 모델": f"{deg}차 곡선 ({'직선' if deg == 1 else '다항선'})",
        "테스트 데이터 MAE (평균 오차)": f"{mae:.2f} °C",
        "2050년 예상 기온": f"{pred_2050:.2f} °C"
    })

# 4. 평가 결과 표 출력
st.subheader("📊 테스트 데이터 채점 및 2050년 예측 결과")
st.dataframe(pd.DataFrame(eval_results), use_container_width=True)

# 5. Plotly 시각화 (산점도 + 회귀 곡선들)
st.subheader("📈 회귀 곡선 추이 비교")
fig = go.Figure()

# 훈련 데이터 산점도
fig.add_trace(go.Scatter(
    x=train_df["연도"], y=train_df["mean_temp"],
    mode="markers", name="훈련 데이터 (1908~2004)",
    marker=dict(color="#1f77b4", size=6, opacity=0.7)
))

# 테스트 데이터 산점도
fig.add_trace(go.Scatter(
    x=test_df["연도"], y=test_df["mean_temp"],
    mode="markers", name="테스트 데이터 (2005~2025)",
    marker=dict(color="#d62728", size=8, symbol="diamond")
))

# 연속적인 연도 라인 생성 (1908년 ~ 2050년)
plot_years = np.linspace(1908, 2050, 300)
plot_X = pd.DataFrame({"X_scaled": (plot_years - 1908) / 100.0})

colors = ["#2ca02c", "#ff7f0e", "#9467bd"]
line_styles = ["solid", "dash", "dot"]

for deg, color, style in zip(degrees, colors, line_styles):
    plot_y = models[deg].predict(plot_X)
    fig.add_trace(go.Scatter(
        x=plot_years, y=plot_y,
        mode="lines",
        name=f"{deg}차 곡선 예측선",
        line=dict(color=color, width=2.5, dash=style)
    ))

# 2050년 구분
