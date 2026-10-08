import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("📊 학습 기간별 회귀 모델 성능 비교")

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
    
    valid_data = yearly_summary[yearly_summary["count"] >= 300].copy()
    valid_data["X"] = valid_data["연도"] - 1908
    return valid_data

data = load_data()

# 2. 데이터셋 분할 (테스트: 최근 20년 2006~2025)
test_df = data[(data["연도"] >= 2006) & (data["연도"] <= 2025)]
train_50 = data[(data["연도"] >= 1956) & (data["연도"] <= 2005)]
train_100 = data[(data["연도"] >= 1906) & (data["연도"] <= 2005)]
train_full = data.copy()

# 3. 모델 학습 및 평가 함수
def train_and_evaluate(train_set, test_set, model_name):
    X_train = train_set[["X"]]
    y_train = train_set["mean_temp"]
    X_test = test_set[["X"]]
    y_test = test_set["mean_temp"]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    slope = model.coef_[0]
    
    return {
        "모델": model_name,
        "학습 기간": f"{train_set['연도'].min()}~{train_set['연도'].max()}",
        "기울기 (°C/10년)": round(slope * 10, 4),
        "MAE (°C)": round(mae, 4),
        "MSE (°C²)": round(mse, 4),
        "R²": round(r2, 4),
        "model_obj": model
    }

results = [
    train_and_evaluate(train_full, test_df, "전체 데이터 학습"),
    train_and_evaluate(train_100, test_df, "최근 100년 학습 (1906~2005)"),
    train_and_evaluate(train_50, test_df, "최근 50년 학습 (1956~2005)")
]

res_df = pd.DataFrame(results).drop(columns=["model_obj"])

# 4. Streamlit 화면 출력
st.subheader("📋 공통 테스트 데이터(2006~2025) 평가 결과")
st.dataframe(res_df, use_container_width=True)

st.subheader("📈 회귀선 비교 시각화")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data["연도"], y=data["mean_temp"],
    mode="markers", name="실제 기온 데이터",
    marker=dict(color="gray", opacity=0.5, size=6)
))

fig.add_vrect(
    x0=2006, x1=2025, fillcolor="LightSalmon", opacity=0.2,
    layer="below", line_width=0, annotation_text="테스트 구간 (2006~2025)"
)

x_range = np.arange(1900, 2026)
X_range_df = pd.DataFrame({"X": x_range - 1908})
colors = ["#1f77b4", "#2ca02c", "#d62728"]

for res, color in zip(results, colors):
    y_line = res["model_obj"].predict(X_range_df)
    fig.add_trace(go.Scatter(
        x=x_range, y=y_line,
        mode="lines",
        name=f"{res['모델']} ({res['기울기 (°C/10년)']}°C/10년)",
        line=dict(color=color, width=2)
    ))

fig.update_layout(
    xaxis_title="연도", yaxis_title="평균기온 (°C)",
    template="plotly_white", hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
