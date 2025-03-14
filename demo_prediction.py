import streamlit as st
import pandas as pd


def display_demo_prediction():
    

    
    st.title ("แนวการพัฒนา Machine Learning ")
    st.markdown("แหล่งที่มาของชุดข้อมูล air_quality_modified.csv")
    st.markdown("https://chatgpt.com/")


    st.markdown("#### การจัดเตรียมข้อมูล")

    st.markdown("การโหลดข้อมูล")
    st.code("data = pd.read_csv('data/air_quality_modified.csv')", language='python')

    st.markdown("การแปลงคอลัมน์")
    st.code("data['Date'] = pd.to_datetime(data['Date'])", language='python')

    st.markdown("กรองข้อมูลในช่วงเวลา 5 ปีล่าสุด")
    code='''end_date = data['Date'].max()
start_date = end_date - pd.DateOffset(years=5)
data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
data.set_index('Date', inplace=True)
'''
    st.code(code, language='python')

    st.markdown("การจัดการค่าที่หายไป (Missing Values)")
    code='''imputer = SimpleImputer(strategy='mean')
y = imputer.fit_transform(y.values.reshape(-1, 1)).flatten()
X = imputer.fit_transform(X)
'''
    st.code(code, language='python')

    st.markdown("#### ทฤษฎีของอัลกอริทึมที่พัฒนา")

    st.markdown("Neural Network (NN) เนื่องจากมีความสามารถในการจัดการกับข้อมูลที่ซับซ้อนและไม่เป็นเชิงเส้น, สามารถเรียนรู้จากข้อมูลที่มีหลายปัจจัย, และปรับแต่งพารามิเตอร์ได้ตามความเหมาะสม. นอกจากนี้ยังสามารถจัดการกับข้อมูลที่มีค่าหายไป (missing values) และทำงานได้ดีในข้อมูลที่มีมิติสูงหรือข้อมูลขนาดใหญ่")



    st.markdown("#### ขั้นตอนการพัฒนาโมเดล")
    st.markdown("การสร้างโมเดล")
    st.code("model_nn = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, activation='relu', solver='adam', random_state=42)", language='python')

    st.markdown("การฝึกโมเดล")
    st.code("model_nn.fit(X_train, y_train)", language='python')

    
    st.markdown("การทำนายค่าจากโมเดล")
    st.code("y_pred = model.predict(X_test)", language='python')

    
    st.markdown("การแสดงกราฟ")
    code='''plt.plot(dates, actual, color='lightblue', label='Actual', linestyle='-', linewidth=2)
plt.plot(dates, predicted, color='red', label='Predicted', linestyle='--', linewidth=2)
'''
    st.code(code, language='python')

    
   

    
    

    

    