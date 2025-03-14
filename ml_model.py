import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def display_ml_models():
    st.title("Machine Learning Model ทำนายคุณภาพอากาศ")
    
    if st.button('แสดงกราฟ'):
        try:
            data = pd.read_csv('data/air_quality_100_years (1).csv')  # ปรับให้ตรงกับเส้นทางไฟล์ของคุณ
        except FileNotFoundError:
            st.error("ไม่พบไฟล์ 'air_quality_100_years (1).csv' กรุณาตรวจสอบเส้นทางไฟล์")
            return
        
        # แปลงคอลัมน์ 'Date' เป็น datetime
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        
        # กรองข้อมูลที่มีวันที่ 5 ปีย้อนหลังจากปัจจุบัน
        end_date = pd.to_datetime('today')
        start_date = end_date - pd.DateOffset(years=5)
        data_filtered = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        
        # เติมค่าที่หายไปในตัวแปร AQI โดยใช้ SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        data_imputed = data_filtered.copy()  # สร้างสำเนาของข้อมูล
        data_imputed['AQI'] = imputer.fit_transform(data_filtered[['AQI']])  # เติมค่า NaN ใน 'AQI'

        # แยกข้อมูลเป็น X (features) และ y (target)
        X = data_imputed.drop(['Date', 'AQI'], axis=1)  # ลบคอลัมน์ที่ไม่ใช้
        y = data_imputed['AQI']

        # เติมค่า NaN ใน X ก่อนการแบ่งข้อมูล
        X = imputer.fit_transform(X)  # ใช้ SimpleImputer เติมค่า NaN ใน X

        # การปรับขนาดข้อมูล (Standardization)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # แบ่งข้อมูลเป็น Train และ Test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # โมเดลที่ 1: Decision Tree
        dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)
        mse_dt = mean_squared_error(y_test, y_pred_dt)
        r2_dt = r2_score(y_test, y_pred_dt)
        
        # โมเดลที่ 2: K-Nearest Neighbors (KNN) พร้อม GridSearchCV
        knn_model = KNeighborsRegressor()
        knn_params = {'n_neighbors': [3, 5, 7, 10, 15]}
        knn_grid_search = GridSearchCV(knn_model, knn_params, cv=5, scoring='neg_mean_squared_error')
        knn_grid_search.fit(X_train, y_train)
        best_knn_model = knn_grid_search.best_estimator_
        y_pred_knn = best_knn_model.predict(X_test)
        mse_knn = mean_squared_error(y_test, y_pred_knn)
        r2_knn = r2_score(y_test, y_pred_knn)
        
        # โมเดลที่ 3: Support Vector Regressor (SVR) พร้อม GridSearchCV
        svr_model = SVR()
        svr_params = {'C': [1, 5, 10], 'epsilon': [0.01, 0.05, 0.1]}
        svr_grid_search = GridSearchCV(svr_model, svr_params, cv=5, scoring='neg_mean_squared_error')
        svr_grid_search.fit(X_train, y_train)
        best_svr_model = svr_grid_search.best_estimator_
        y_pred_svr = best_svr_model.predict(X_test)
        mse_svr = mean_squared_error(y_test, y_pred_svr)
        r2_svr = r2_score(y_test, y_pred_svr)

        # แสดงผลการประเมินผลโมเดล KNN และ SVR (ลบการแสดงผลสำหรับ Decision Tree)
        st.write(f'Mean Squared Error ของโมเดล KNN: {mse_knn}')
        st.write(f'Mean Squared Error ของโมเดล SVR: {mse_svr}')
        st.write(f'Mean Squared Error ของโมเดล Decision Tree: {mse_dt}')

        # แสดงกราฟที่แสดงค่าจริง vs ค่าทำนาย
        fig, axs = plt.subplots(3, 1, figsize=(10, 18))  # ปรับเป็น 3 กราฟสำหรับ Decision Tree, KNN, SVR
        
        # กราฟ Decision Tree
        axs[0].plot(y_test.values, label='Actual', color='blue', linestyle='-', linewidth=2)
        axs[0].plot(y_pred_dt, label='Predicted', color='red', linestyle='--', linewidth=2)
        axs[0].set_title('Decision Tree: Actual vs Predicted')
        axs[0].set_xlabel('Sample Index')
        axs[0].set_ylabel('Air Quality Index (AQI)')
        axs[0].legend()
        axs[0].grid(True)

        # กราฟ KNN
        axs[1].plot(y_test.values, label='Actual', color='blue', linestyle='-', linewidth=2)
        axs[1].plot(y_pred_knn, label='Predicted', color='red', linestyle='--', linewidth=2)
        axs[1].set_title('KNN: Actual vs Predicted')
        axs[1].set_xlabel('Sample Index')
        axs[1].set_ylabel('Air Quality Index (AQI)')
        axs[1].legend()
        axs[1].grid(True)

        # กราฟ SVR
        axs[2].plot(y_test.values, label='Actual', color='blue', linestyle='-', linewidth=2)
        axs[2].plot(y_pred_svr, label='Predicted', color='red', linestyle='--', linewidth=2)
        axs[2].set_title('SVR: Actual vs Predicted')
        axs[2].set_xlabel('Sample Index')
        axs[2].set_ylabel('Air Quality Index (AQI)')
        axs[2].legend()
        axs[2].grid(True)

        st.pyplot(fig)  # แสดงกราฟทั้งหมด

# เรียกใช้ฟังก์ชันใน Streamlit
if __name__ == "__main__":
    display_ml_models()
