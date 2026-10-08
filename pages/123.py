import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 데이터 불러오기 및 전처리
url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
df = pd.read_csv(url, encoding="utf-8")

df["날짜"] = pd.to_datetime(df["날짜"])
df["연도"] = df["날짜"].dt.year

# 2025년 이하 & 관측일수 300일 이상 필터링
df_filtered = df[df["연도"] <= 2025]
yearly_summary = df_filtered.groupby("연도")["평균기온"].agg(
    count="count",
    mean_temp="mean"
).reset_index()

data = yearly_summary[yearly_summary["count"] >= 300].copy()

# 독립변수 X: 1908년부터 지난 연수
data["X"] = data["연도"] - 1908

# 2. 데이터셋 분할
# 공통 테스트 데이터 (최근 20년: 2006 ~ 2025)
test_df = data[(data["연도"] >= 2006) & (data["연도"] <= 2025)]

# 훈련 데이터 1: 최근 50년 (1956 ~ 2005)
train_50 = data[(data["연도"] >= 1956) & (data["연도"] <= 2005)]

# 훈련 데이터 2: 최근 100년 (1906 ~ 2005)
train_100 = data[(data["연도"] >= 1906) & (data["연도"] <= 2005)]

# 전체 데이터 (비교용)
train_full = data.copy()

# 3. 모델 학습 및 평가 함수
def train_and_evaluate(train_set, test_set, model_name):
    X_train = train_set[["X"]]
    y_train = train_set["mean_temp"]
    
    X_test = test_set[["X"]]
    y_test = test_set["mean_temp"]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 테스트 데이터 예측
    y_pred = model.predict(X_test)
    
    # 평가 지표 계산
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    slope = model.coef_[0]          # 연간 기온 상승폭 (°C/년)
    intercept = model.intercept_
    
    return {
        "모델": model_name,
        "학습 기간": f"{train_set['연도'].min()}~{train_set['연도'].max()} ({len(train_set)}개 해)",
        "기울기 (°C/10년)": round(slope * 10, 4),
        "절편": round(intercept, 4),
        "MAE (°C)": round(mae, 4),
        "MSE (°C²)": round(mse, 4),
        "R²": round(r2, 4),
        "model_obj": model
    }

# 4. 각 모델 평가 실행
results = []
results.append(train_and_evaluate(train_full, test_df, "전체 데이터 학습 모델"))
results.append(train_and_evaluate(train_100, test_df, "최근 100년 학습 모델 (1906~2005)"))
results.append(train_and_evaluate(train_50, test_df, "최근 50년 학습 모델 (1956~2005)"))

res_df = pd.DataFrame(results).drop(columns=["model_obj"])
print("=== 모델별 테스트 데이터(2006~2025) 예측 성능 비교 ===")
print(res_df.to_string(index=False))

# 5. Plotly 회귀선 비교 시각화
fig = go.Figure()

# 실제 관측 데이터 (산점도)
fig.add_trace(go.Scatter(
    x=data["연도"], y=data["mean_temp"],
    mode="markers", name="실제 기온 데이터",
    marker=dict(color="gray", opacity=0.6, size=6)
))

# 테스트 영역 강조 표시
fig.add_vrect(
    x0=2006, x1=2025, fillcolor="LightSalmon", opacity=0.2,
    layer="below", line_width=0, annotation_text="공통 테스트 구간 (2006~2025)"
)

# 회귀선 그리기 (1900년 ~ 2025년)
x_range = np.arange(1900, 2026)
X_range_df = pd.DataFrame({"X": x_range - 1908})

colors = ["#1f77b4", "#2ca02c", "#d62728"]
for res, color in zip(results, colors):
    y_line = res["model_obj"].predict(X_range_df)
    fig.add_trace(go.Scatter(
        x=x_range, y=y_line,
        mode="lines", name=f"{res['모델']} (10년당 {res['기울기 (°C/10년)']}°C)",
        line=dict(color=color, width=2)
    ))

fig.update_layout(
    title="학습 기간별 서울 연평균 기온 회귀선 및 테스트 데이터(2006~2025) 비교",
    xaxis_title="연도", yaxis_title="평균기온 (°C)",
    template="plotly_white", hovermode="x unified"
)
fig.show()
