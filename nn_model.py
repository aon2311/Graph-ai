import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def display_nn_model():
    st.title("Neural Network Model ทำนายคุณภาพอากาศ")

    # ฟังก์ชันสำหรับสร้างและฝึกโมเดล Neural Network
    def build_and_train_model(X_train, y_train):
        # สร้าง Neural Network Model (MLPRegressor จาก Scikit-learn) พร้อมการตั้งค่า Hyperparameters
        model_nn = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, activation='relu', solver='adam', random_state=42)
        # ฝึกโมเดล
        model_nn.fit(X_train, y_train)
        return model_nn

    # ฟังก์ชันสำหรับการประเมินผลโมเดล
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        return y_pred

    # ฟังก์ชันแสดงกราฟเส้น (Actual vs Predicted)
    def plot_line(actual, predicted, dates, y_label='PM2.5'):
        # ขนาดกราฟ
        plt.figure(figsize=(10, 6))  
        plt.plot(dates, actual, color='lightblue', label='Actual', linestyle='-', linewidth=2)  # เปลี่ยนจาก 'blue' เป็น 'lightblue'
        plt.plot(dates, predicted, color='red', label='Predicted', linestyle='--', linewidth=2)  # กราฟค่าคาดการณ์
        plt.title(f'Actual vs Predicted {y_label} Values Over Time')  # ชื่อกราฟ
        plt.xlabel('Date')  # ชื่อแกน X (เป็น Date)
        plt.ylabel(y_label)  # ชื่อแกน Y
        plt.legend()  # แสดง legend
        plt.grid(True)  # แสดงกริด

        # แสดงทุกๆ 1 เดือน (เลือกแสดงต้นเดือน)
        month_range = pd.date_range(dates.min(), dates.max(), freq='MS')  # เลือกทุกๆ 1 เดือน (ต้นเดือน)
        plt.xticks(month_range, rotation=45)  # หมุนค่าปีให้สามารถอ่านได้
        st.pyplot(plt)  # แสดงกราฟ

    # ปุ่มให้ผู้ใช้กดเพื่อแสดงกราฟ
    if st.button('แสดงกราฟ'):
        # โหลดข้อมูลจากไฟล์ air_quality_modified.csv ที่อยู่ในโฟลเดอร์ data
        try:
            data = pd.read_csv('data/air_quality_modified.csv')  # โหลดข้อมูลจากไฟล์ในโฟลเดอร์ data
        except FileNotFoundError:
            st.error("ไม่พบไฟล์ 'air_quality_modified.csv' กรุณาตรวจสอบเส้นทางไฟล์")
            return

        # ตรวจสอบว่ามีคอลัมน์ 'Date' หรือไม่
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])  # แปลงคอลัมน์ 'Date' เป็น datetime

            # เลือกข้อมูล 5 ปีล่าสุด (จากปีปัจจุบัน)
            end_date = data['Date'].max()  # วันที่ล่าสุดในข้อมูล
            start_date = end_date - pd.DateOffset(years=5)  # วันที่เริ่มต้นคือ 5 ปีที่แล้ว

            # กรองข้อมูลที่มีช่วงเวลา 5 ปีล่าสุด
            data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

            data.set_index('Date', inplace=True)  # ตั้งค่า Date เป็น index

        # คัดเลือกคอลัมน์ที่จะใช้
        if 'PM2.5' in data.columns:  # ใช้คอลัมน์ 'PM2.5' เป็น target
            # สมมติว่าใช้คอลัมน์อื่น ๆ ในข้อมูลเป็น feature
            X = data.drop(['PM2.5'], axis=1)  # คอลัมน์ที่ไม่ใช้เป็น feature
            y = data['PM2.5']  # 'PM2.5' ใช้เป็น target

            # ตรวจสอบและจัดการค่า NaN ใน y
            if y.isna().sum() > 0:
                st.warning(f"พบค่า NaN ใน target variable (PM2.5), กำลังเติมค่า NaN ด้วยค่าเฉลี่ย")
                # เติม NaN ใน target variable ด้วยค่าเฉลี่ย
                imputer = SimpleImputer(strategy='mean')
                y = imputer.fit_transform(y.values.reshape(-1, 1)).flatten()

            # ใช้ SimpleImputer เพื่อเติมค่า NaN ใน X (feature)
            imputer = SimpleImputer(strategy='mean')  # เติมค่า NaN ด้วยค่าเฉลี่ย
            X = imputer.fit_transform(X)  # แทนที่ค่า NaN ใน X ด้วยค่าเฉลี่ย

            # แบ่งข้อมูลเป็น train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # ปรับสเกลข้อมูล
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # สร้างและฝึกโมเดล Neural Network
            model_nn = build_and_train_model(X_train_scaled, y_train)

            # ทำนายค่าจากโมเดล
            y_pred = evaluate_model(model_nn, X_test_scaled, y_test)

            # สร้าง index ที่ตรงกันกับข้อมูล y_test โดยใช้ index ของ data ที่ตรงกับ y_test
            dates = data.index[-len(y_test):]  # ใช้ index ของ data ที่ตรงกับ y_test

            # ตรวจสอบให้แน่ใจว่า y_pred และ y_test มีขนาดตรงกัน
            if len(dates) != len(y_test):
                st.error(f"ขนาดของ dates และ y_test ไม่ตรงกัน: {len(dates)} vs {len(y_test)}")
            else:
                plot_line(y_test, y_pred, dates)  # แสดงกราฟระหว่างค่าจริงกับค่าทำนาย
        else:
            st.error("ไม่พบคอลัมน์ 'PM2.5' ในข้อมูล กรุณาตรวจสอบไฟล์ CSV")