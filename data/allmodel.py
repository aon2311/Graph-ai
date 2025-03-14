import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# กำหนดเส้นทางไฟล์
file_name = os.path.join('..', 'data', 'air_quality_modified.csv')

# ตรวจสอบว่าไฟล์มีอยู่หรือไม่
if os.path.exists(file_name):
    data = pd.read_csv(file_name)
else:
    raise FileNotFoundError(f"ไฟล์ {file_name} ไม่พบในตำแหน่งที่กำหนด. โปรดตรวจสอบเส้นทางไฟล์.")

# ตรวจสอบข้อมูลเบื้องต้น
print(data.info())
print(data.isnull().sum())  # ตรวจสอบค่าที่หายไป

# แยกคอลัมน์ 'Date' ออกจากข้อมูลตัวเลข
date_column = data['Date']
data_numeric = data.drop(['Date'], axis=1)

# การเติมค่าที่หายไปโดยใช้ SimpleImputer
imputer = SimpleImputer(strategy='mean')  # ใช้ค่าเฉลี่ยในการเติมค่าที่หายไป
data_imputed = pd.DataFrame(imputer.fit_transform(data_numeric), columns=data_numeric.columns)

# รวมคอลัมน์ Date กลับเข้ามา
data_imputed['Date'] = date_column

# แยกข้อมูลเป็น X (features) และ y (target)
X = data_imputed.drop(['AQI', 'Date'], axis=1)  # ลบคอลัมน์ที่ไม่ต้องการ
y_temp = data_imputed['AQI']  # เปลี่ยนจาก 'Air_Quality' เป็น 'AQI'

# แบ่งข้อมูลเป็น Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)

# Model 1: Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f'Decision Tree Mean Squared Error: {mse_dt}')
cv_scores_dt = cross_val_score(dt_model, X, y_temp, cv=5, scoring='neg_mean_squared_error')
print(f'Decision Tree Cross-validation MSE: {-cv_scores_dt.mean()}')
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', color='blue', linestyle='-', linewidth=2)
plt.plot(y_pred_dt, label='Predicted', color='red', linestyle='--', linewidth=2)
plt.title('Decision Tree: Actual vs Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Air Quality')
plt.legend()
plt.grid(True)

# Model 2: K-Nearest Neighbors (KNN) Regressor
knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f'KNN Mean Squared Error: {mse_knn}')
cv_scores_knn = cross_val_score(knn, X, y_temp, cv=5, scoring='neg_mean_squared_error')
print(f'KNN Cross-validation MSE: {-cv_scores_knn.mean()}')

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue', linewidth=2)
plt.plot(y_test.index, y_pred_knn, label='Predicted', color='red', linestyle='--', linewidth=2)
plt.xlabel('Index')
plt.ylabel('Air Quality (AQI)')
plt.title('KNN Regressor: Actual vs Predicted')
plt.legend()
plt.show()

# Model 3: Support Vector Regressor (SVR)
def preprocess_data(X, y):
    num_imputer = SimpleImputer(strategy='mean')
    X_numeric_imputed = pd.DataFrame(num_imputer.fit_transform(X), columns=X.columns)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_numeric_imputed), columns=X.columns)
    return X_scaled, y, scaler

X_scaled, y, scaler = preprocess_data(X, y_temp)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svr_model = SVR(kernel='rbf', C=5.0, epsilon=0.05)
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2 = r2_score(y_test, y_pred_svr)
mape = np.mean(np.abs((y_test - y_pred_svr) / y_test)) * 100
accuracy = 100 - mape

print(f"SVR Model Evaluation:")
print(f"MAE: {mae}")
print(f"MSE: {mse_svr}")
print(f"R^2 Score: {r2}")
print(f"MAPE: {mape}%")
print(f"Prediction Accuracy: {accuracy}%")

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='True Values', color='blue')
plt.plot(y_pred_svr, label='Predicted Values', color='red')
plt.legend()
plt.title('SVR Model: True vs Predicted')
plt.xlabel('Index')
plt.ylabel('PM2.5')
plt.show()

# บันทึกโมเดลทั้งหมดในไฟล์เดียวกัน
models = {
    'DecisionTree': dt_model,
    'KNN': knn,
    'SVR': svr_model,
    'Scaler': scaler
}

# บันทึกโมเดลทั้งหมดเป็นไฟล์เดียว
joblib.dump(models, 'all_models.pkl')
print("All models saved in 'all_models.pkl'")
