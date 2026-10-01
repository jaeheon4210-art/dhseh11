import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 설정 및 제목
st.set_page_config(page_title="서울 기온 예측기", page_icon="🌡️", layout="wide")

st.title("🌡️ 서울 연평균기온 예측기")
st.write("서울의 기후 데이터를 바탕으로 과거 평균기온 추세를 확인하고, 전체 기간 및 최근 20년의 상승 속도를 비교합니다.")

# 2. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data(url):
    # UTF-8 인코딩으로 데이터 로드
    df = pd.read_csv(url, encoding='utf-8')
    
    # 날짜 컬럼을 datetime 형식으로 변환 및 연도 추출
    df['날짜'] = pd.to_datetime(df['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    # 2025년 이하 데이터 및 평균기온 결측치 제거
    df_valid = df[(df['연도'] <= 2025) & df['평균기온'].notna()].copy()
    
    # 연도별 관측일 수 계산 후 300일 미만 연도 제외
    year_counts = df_valid.groupby('연도')['평균기온'].count()
    valid_years = year_counts[year_counts >= 300].index
    
    df_filtered = df_valid[df_valid['연도'].isin(valid_years)]
    
    # 연도별 평균기온 계산
    df_yearly = df_filtered.groupby('연도')['평균기온'].mean().reset_index()
    return df_yearly

df_yearly = load_and_process_data(DATA_URL)

# 독립 변수 X: 1908년부터 지난 연수
df_yearly['X'] = df_yearly['연도'] - 1908

# 3. 회귀 모델 계산 (전체 기간 vs 최근 20년)
# (1) 전체 기간 회귀
slope_all, intercept_all = np.polyfit(df_yearly['X'], df_yearly['평균기온'], 1)
corr_all = df_yearly['연도'].corr(df_yearly['평균기온'])
slope_100_all = slope_all * 100  # 100년당 상승 도수

start_year_all = int(df_yearly['연도'].min())
end_year_all = int(df_yearly['연도'].max())
total_years_all = len(df_yearly)

# (2) 최근 20년 회귀
df_recent = df_yearly.tail(20).copy()
slope_recent, intercept_recent = np.polyfit(df_recent['X'], df_recent['평균기온'], 1)
corr_recent = df_recent['연도'].corr(df_recent['평균기온'])
slope_100_recent = slope_recent * 100  # 100년당 상승 도수

start_year_recent = int(df_recent['연도'].min())
end_year_recent = int(df_recent['연도'].max())
total_years_recent = len(df_recent)

# 4. 분석 개요 정보
col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("분석 시작 연도", f"{start_year_all}년")
col_info2.metric("분석 끝 연도", f"{end_year_all}년")
col_info3.metric("분석 대상 해의 개수", f"{total_years_all}개")

st.divider()

# 5. 기울기 비교 (100년에 몇 도 오르는가)
st.subheader("🔥 기온 상승 속도 비교 (100년 기준)")

col_comp1, col_comp2 = st.columns(2)

with col_comp1:
    st.markdown("### 🌐 전체 기간")
    st.caption(f"기준: {start_year_all}년 ~ {end_year_all}년 ({total_years_all}개 연도)")
    st.metric(
        label="100년당 기온 상승 폭",
        value=f"+{slope_100_all:.2f} °C / 100년",
        delta=f"1년당 약 +{slope_all:.4f} °C"
    )
    st.write(f"- 상관계수: **{corr_all:.4f}**")

with col_comp2:
    st.markdown("### ⚡ 최근 20년")
    st.caption(f"기준: {start_year_recent}년 ~ {end_year_recent}년 ({total_years_recent}개 연도)")
    
    diff_slope_100 = slope_100_recent - slope_100_all
    st.metric(
        label="100년당 기온 상승 폭",
        value=f"+{slope_100_recent:.2f} °C / 100년",
        delta=f"전체 기간 대비 {diff_slope_100:+.2f} °C / 100년"
    )
    st.write(f"- 상관계수: **{corr_recent:.4f}**")

st.divider()

# 6. 연도 선택 및 예상 기온 예측
st.subheader("🔮 예상 기온 예측")
selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 예측 기온 계산
predicted_temp_all = slope_all * (selected_year - 1908) + intercept_all
predicted_temp_recent = slope_recent * (selected_year - 1908) + intercept_recent

pred_col1, pred_col2 = st.columns(2)
pred_col1.metric(
    label=f"📌 {selected_year}년 예상 기온 (전체 기간 모델)",
    value=f"{predicted_temp_all:.2f} °C"
)
pred_col2.metric(
    label=f"📌 {selected_year}년 예상 기온 (최근 20년 모델)",
    value=f"{predicted_temp_recent:.2f} °C"
)

st.divider()

# 7. Plotly 시각화 (산점도 및 두 회귀선)
st.subheader("📊 연도별 평균기온 추이 및 회귀 직선")

# 회귀선 x축 범위
min_x_year = min(start_year_all, 1900)
max_x_year = max(end_year_all, selected_year)
plot_years = np.arange(min_x_year, max_x_year + 1)

plot_y_all = slope_all * (plot_years - 1908) + intercept_all
plot_y_recent = slope_recent * (plot_years - 1908) + intercept_recent

fig = go.Figure()

# 실제 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=df_yearly['연도'],
    y=df_yearly['평균기온'],
    mode='markers',
    name='실제 연평균기온',
    marker=dict(color='#2b5c8f', size=7)
))

# 전체 기간 회귀선
fig.add_trace(go.Scatter(
    x=plot_years,
    y=plot_y_all,
    mode='lines',
    name=f'전체 기간 회귀선 (+{slope_100_all:.2f}°C/100년)',
    line=dict(color='#d9534f', width=2.5, dash='dash')
))

# 최근 20년 회귀선
fig.add_trace(go.Scatter(
    x=plot_years,
    y=plot_y_recent,
    mode='lines',
    name=f'최근 20년 회귀선 (+{slope_100_recent:.2f}°C/100년)',
    line=dict(color='#f0ad4e', width=2.5, dash='dot')
))

# 현재 선택된 연도의 전체 기간 예측 위치 표시
fig.add_trace(go.Scatter(
    x=[selected_year],
    y=[predicted_temp_all],
    mode='markers+text',
    name=f'{selected_year}년 예측점 (전체 모델)',
    marker=dict(color='#d9534f', size=13, symbol='star'),
    text=[f"{predicted_temp_all:.2f}°C"],
    textposition="top center"
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    height=550
)

st.plotly_chart(fig, use_container_width=True)
