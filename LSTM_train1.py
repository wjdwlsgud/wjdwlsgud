import random
import os
import numpy as np
import tensorflow as tf

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # TensorFlow 결정론적 연산 활성화 (GPU 사용 시 필요)
    tf.config.experimental.enable_op_determinism()

# 시드 설정 함수 호출
set_seed(42)




import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# -------------------- 1) 데이터 로드 & 병합 --------------------
demand_df = pd.read_csv(r"C:\vscode\demand\demand.csv", encoding='utf-8')
weather_df = pd.read_csv(r"C:\vscode\demand\reference\asos_weighted_2021_2024_filled.csv", encoding='utf-8')

demand_df['datetime'] = pd.to_datetime(demand_df['datetime'])
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
df = pd.merge(demand_df, weather_df, on='datetime', how='inner')


df.set_index('datetime', inplace=True)
df = df.sort_index()
df = df[~df.index.duplicated(keep='first')]

df['hour'] = df.index.hour

for cat_col in ['datetype', 'holiday']:
    if cat_col in df.columns and not np.issubdtype(df[cat_col].dtype, np.number):
        df[cat_col] = df[cat_col].astype('category').cat.codes

# 과거 수요 지연특성
df['demand_t48']  = df['demand'].shift(48)
df['demand_t168'] = df['demand'].shift(168)

# shift로 생긴 NaN 제거 후 슬라이스
df = df.dropna()

train_df = df.loc['2021-01-01':'2023-12-31'].copy()
test_df  = df.loc['2024-01-01':'2024-12-31'].copy()

feature_cols = [
    'temp_C', 'humidity_pct', 'wind_ms', 'irr', 'precip_mm', 'hour', 'datetype', 'holiday',
    'temp_sens_pct_perC', 'demand_t48', 'demand_t168'
]
target_col = 'demand'

X_train = train_df[feature_cols].to_numpy()
y_train = train_df[target_col].to_numpy()
X_test  = test_df[feature_cols].to_numpy()
y_test  = test_df[target_col].to_numpy()

# -------------------- 3) 스케일링 --------------------
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled  = scaler_y.transform(y_test.reshape(-1, 1))

# -------------------- 4) 시퀀스 생성 (과거47 + y시점X 1 = 48, y는 48시간 뒤) --------------------
def create_sequences_gap_with_futureX(x_data, y_data, past_len, gap_hours):
    """
    X: (past_len 스텝) + (y시점의 X 1스텝) = 총 past_len+1 스텝
    y: (입력 마지막 과거 시점으로부터 gap_hours 뒤) 타깃 1개
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data).reshape(-1)
    T, F = x_data.shape
    X, Y = [], []
    # i: 과거블록의 마지막 인덱스
    for i in range(past_len-1, T - gap_hours):
        y_idx = i + gap_hours
        X_past   = x_data[i - (past_len - 1) : i + 1]   # (past_len, F)
        X_future = x_data[y_idx : y_idx + 1]            # (1, F)  ← y 시점의 X
        X_seq = np.vstack([X_past, X_future])           # (past_len+1, F)
        X.append(X_seq)
        Y.append(y_data[y_idx])                          # y 시점의 y
    return np.array(X), np.array(Y).reshape(-1, 1)

SQU_LENGTH = 48          # 최종 입력 타임스텝 수(= 과거47 + y시점X 1)
PAST_LEN   = SQU_LENGTH - 1
GAP_HOURS  = 48          # 과거블록 마지막 시점으로부터 y까지의 갭

# y는 스케일 상태로 시퀀스 구성 (학습 안정)
X_train_seq, y_train_seq = create_sequences_gap_with_futureX(
    X_train_scaled, y_train_scaled, past_len=PAST_LEN, gap_hours=GAP_HOURS
)
X_test_seq,  y_test_seq  = create_sequences_gap_with_futureX(
    X_test_scaled,  y_test_scaled,  past_len=PAST_LEN, gap_hours=GAP_HOURS
)

print(f"훈련 데이터 형태: X={X_train_seq.shape}, y={y_train_seq.shape}")
print(f"테스트 데이터 형태: X={X_test_seq.shape}, y={y_test_seq.shape}")

# -------------------- 5) 모델 & EarlyStopping --------------------
print("\n5. LSTM 모델 구축 및 훈련 중...")
model = Sequential([
    LSTM(64, activation='relu', input_shape=(SQU_LENGTH, X_train_seq.shape[2])),
    Dense(1)  # 1시간 타깃
])
model.compile(optimizer='adam', loss='mean_squared_error')

es = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    min_delta=0.0,
    verbose=1
)

history = model.fit(
    X_train_seq, y_train_seq,
    epochs=200,
    batch_size=64,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# -------------------- 6) 손실 곡선 --------------------
plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

# -------------------- 7) 예측 & 역변환 --------------------
pred_scaled   = model.predict(X_test_seq)
predictions   = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
y_test_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).reshape(-1)

# -------------------- 8) 성능 지표 --------------------
mae_all  = mean_absolute_error(y_test_actual, predictions)
mape_all = mean_absolute_percentage_error(y_test_actual, predictions) * 100
r2_all   = r2_score(y_test_actual, predictions)
print(f"\n=== 전체(2024) 성능 ===\nMAE : {mae_all:.2f}\nMAPE: {mape_all:.2f}%\nR²  : {r2_all:.3f}")

# -------------------- 9) 타임스탬프 정렬 --------------------
# y의 첫 시점 = (PAST_LEN-1) + GAP_HOURS 이 아니라, 위 구현에선 i 시작이 (PAST_LEN-1),
# y_idx = i + GAP_HOURS 이므로 offset = (PAST_LEN-1) + GAP_HOURS
offset = (PAST_LEN - 1) + GAP_HOURS
pred_dates = test_df.index[offset : offset + len(y_test_actual)]

# -------------------- 10) 시각화 --------------------
# (A) 주간(처음 168시간)
plt.figure(figsize=(15,5))
plt.plot(pred_dates[:168], y_test_actual[:168], label='실제 수요')
plt.plot(pred_dates[:168], predictions[:168], label='예측 수요', linestyle='--')
plt.title(f'2024년 1주 구간 예측 (MAE={mae_all:.0f}, MAPE={mape_all:.2f}%)')
plt.xlabel('날짜'); plt.ylabel('전력 수요')
plt.legend(); plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gcf().autofmt_xdate()
plt.tight_layout(); plt.show()

# (B) 2024년 전체
plt.figure(figsize=(15,5))
plt.plot(pred_dates, y_test_actual, label='실제 수요', alpha=0.85)
plt.plot(pred_dates, predictions,   label='예측 수요', linestyle='--', alpha=0.85)
plt.title('2024년 전체 예측 결과')
plt.xlabel('날짜'); plt.ylabel('전력 수요')
plt.legend(); plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
plt.gcf().autofmt_xdate()
plt.tight_layout(); plt.show()

# (C) 패리티 플롯
plt.figure(figsize=(5,5))
plt.scatter(y_test_actual, predictions, s=8, alpha=0.6)
vmin = min(y_test_actual.min(), predictions.min())
vmax = max(y_test_actual.max(), predictions.max())
plt.plot([vmin, vmax], [vmin, vmax], linewidth=2)
plt.title(f'Parity Plot\nMAE={mae_all:.1f}, MAPE={mape_all:.2f}%, R²={r2_all:.3f}')
plt.xlabel('Actual'); plt.ylabel('Predicted')
plt.grid(True); plt.tight_layout(); plt.show()

# -------------------- 11) 월별 지표 --------------------
results_df = pd.DataFrame({
    "datetime": pred_dates,
    "y_true": y_test_actual,
    "y_pred": predictions
})
results_df["month"] = results_df["datetime"].dt.to_period("M")

monthly_metrics = []
for month, group in results_df.groupby("month"):
    y_t = group["y_true"].values
    y_p = group["y_pred"].values
    if len(y_t) < 10:
        continue
    mae_m  = mean_absolute_error(y_t, y_p)
    mape_m = mean_absolute_percentage_error(y_t, y_p) * 100
    r2_m   = r2_score(y_t, y_p)
    monthly_metrics.append({"month": str(month), "MAE": mae_m, "MAPE(%)": mape_m, "R2": r2_m})

monthly_df = pd.DataFrame(monthly_metrics).sort_values("month")
print("\n=== 2024년 월별 성능 지표 ===")
print(monthly_df.round(2))





# -------------------- 12) 그룹별 평가 (월/요일/시간대/공휴일 vs 평상일) --------------------
from collections import OrderedDict

# results_df: ["datetime","y_true","y_pred","month"] 이미 존재
# test_df의 메타(시간, holiday/datetype 등)를 결합해 평가 기준 컬럼 확장
meta_cols = []
if 'hour' in test_df.columns:
    meta_cols.append('hour')

if 'holiday' in test_df.columns:
    meta_cols.append('holiday')

if 'datetype' in test_df.columns:
    meta_cols.append('datetype')

meta = test_df.reset_index()[['datetime'] + meta_cols] if meta_cols else test_df.reset_index()[['datetime']]
eval_df = results_df.merge(meta, on='datetime', how='left')

# hour/weekday/holiday 라벨 만들기
eval_df['hour'] = eval_df['hour'].fillna(eval_df['datetime'].dt.hour).astype(int)
eval_df['weekday_num'] = eval_df['datetime'].dt.weekday   # 0=Mon, 6=Sun
weekday_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
eval_df['weekday'] = eval_df['weekday_num'].map(weekday_map)

# 공휴일 vs 평상일 라벨: holiday 컬럼이 숫자면 >0을 공휴일로 간주, 없거나 전부 NA면 주말을 휴일로 대체
if 'holiday' in eval_df.columns and eval_df['holiday'].notna().any():
    holiday_flag = pd.to_numeric(eval_df['holiday'], errors='coerce').fillna(0) > 0
else:
    holiday_flag = eval_df['weekday_num'] >= 5   # 대체 규칙: 토/일을 휴일로
eval_df['day_type_bin'] = np.where(holiday_flag, 'Holiday', 'Weekday')

def _safe_r2(y_true, y_pred):
    # 분산 0인 경우 R2가 정의되지 않으므로 NaN 처리
    return r2_score(y_true, y_pred) if np.var(y_true) > 0 else np.nan

def _group_metrics(df_group):
    y_t = df_group['y_true'].values
    y_p = df_group['y_pred'].values
    return pd.Series(OrderedDict(
        N = len(y_t),
        MAE = mean_absolute_error(y_t, y_p),
        MAPE_pct = mean_absolute_percentage_error(y_t, y_p) * 100,
        R2 = _safe_r2(y_t, y_p),
    ))

# 12-1) 월별
by_month = eval_df.groupby(eval_df['datetime'].dt.to_period('M')).apply(_group_metrics).reset_index().rename(columns={'datetime':'month'})
by_month = by_month.sort_values('month')
print("\n=== 월별 성능 지표 ===")
print(by_month.round({'MAE':2,'MAPE_pct':2,'R2':3}))

# 12-2) 요일별 (Mon~Sun)
by_weekday = eval_df.groupby('weekday', sort=False).apply(_group_metrics).reset_index()
weekday_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
by_weekday['weekday'] = pd.Categorical(by_weekday['weekday'], categories=weekday_order, ordered=True)
by_weekday = by_weekday.sort_values('weekday')
print("\n=== 요일별 성능 지표 ===")
print(by_weekday.round({'MAE':2,'MAPE_pct':2,'R2':3}))

# 12-3) 시간대별 (0~23시)
by_hour = eval_df.groupby('hour').apply(_group_metrics).reset_index().sort_values('hour')
print("\n=== 시간대별 성능 지표 ===")
print(by_hour.round({'MAE':2,'MAPE_pct':2,'R2':3}).to_string(index=False))

# 12-4) 공휴일 vs 평상일
by_holiday = eval_df.groupby('day_type_bin').apply(_group_metrics).reset_index()
by_holiday['day_type_bin'] = pd.Categorical(by_holiday['day_type_bin'], categories=['Weekday','Holiday'], ordered=True)
by_holiday = by_holiday.sort_values('day_type_bin')
print("\n=== 공휴일 vs 평상일 성능 지표 ===")
print(by_holiday.round({'MAE':2,'MAPE_pct':2,'R2':3}))

# (선택) 빠른 시각화 — 시간대/요일별 MAE
plt.figure(figsize=(10,4))
plt.bar(by_hour['hour'].astype(str), by_hour['MAE'])
plt.title('시간대별 MAE'); plt.xlabel('Hour'); plt.ylabel('MAE'); plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout(); plt.show()

plt.figure(figsize=(7,4))
plt.bar(by_weekday['weekday'].astype(str), by_weekday['MAE'])
plt.title('요일별 MAE'); plt.xlabel('Weekday'); plt.ylabel('MAE'); plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout(); plt.show()



# ================== 12-x) 평일/주말/공휴일 3종 구분 평가 ==================
# eval_df: ["datetime","y_true","y_pred","hour","holiday","datetype","weekday_num","weekday"]가 앞에서 준비돼 있다고 가정

# (1) 주말/공휴일 플래그 생성
is_weekend = eval_df['weekday_num'] >= 5  # 토(5), 일(6)

# holiday 컬럼이 있으면 >0을 공휴일로 간주, 없으면 False
if 'holiday' in eval_df.columns and eval_df['holiday'].notna().any():
    is_holiday = pd.to_numeric(eval_df['holiday'], errors='coerce').fillna(0) > 0
else:
    is_holiday = pd.Series(False, index=eval_df.index)

# (2) 3종 라벨: 공휴일 > 주말 > 평일 우선순위
eval_df['day_type_3'] = np.select(
    [is_holiday, is_weekend],
    ['공휴일', '주말'],
    default='평일'
)

# (3) 3종 구분별 성능 지표
by_daytype3 = eval_df.groupby('day_type_3').apply(_group_metrics).reset_index()
# 보기 좋게 정렬: 평일 → 주말 → 공휴일
order3 = pd.Categorical(by_daytype3['day_type_3'], categories=['평일','주말','공휴일'], ordered=True)
by_daytype3 = by_daytype3.assign(day_type_3=order3).sort_values('day_type_3')

print("\n=== 평일 / 주말 / 공휴일 성능 지표 ===")
print(by_daytype3.round({'MAE':2, 'MAPE_pct':2, 'R2':3}))

# (4) 샘플 수 확인 (데이터 분포 체크)
counts_3 = eval_df['day_type_3'].value_counts().reindex(['평일','주말','공휴일'])
print("\n샘플 수(건):")
print(counts_3)

# (5) 간단 시각화 — 3종 MAE
plt.figure(figsize=(6,4))
plt.bar(by_daytype3['day_type_3'].astype(str), by_daytype3['MAE'])
plt.title('평일/주말/공휴일 MAE'); plt.xlabel('구분'); plt.ylabel('MAE')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout(); plt.show()

# (6) CSV 저장
by_daytype3.to_csv('metrics_by_daytype3.csv', index=False)
print("\nSaved: metrics_by_daytype3.csv")
