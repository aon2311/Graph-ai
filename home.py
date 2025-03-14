import streamlit as st


def display_home():

    


    st.title ("แนวการพัฒนา Machine Learning ")


   
    
    
    st.markdown("แหล่งที่มาของชุดข้อมูล air_quality_100_years (1).csv")
    st.markdown("https://chatgpt.com/")

    st.markdown("#### การจัดเตรียมข้อมูล")

    st.markdown("การโหลดข้อมูล")
    st.code("data = pd.read_csv('data/air_quality_100_years (1).csv')", language='python')

    st.markdown("แปลงวันที่")
    st.code("data['Date'] = pd.to_datetime(data['Date'], errors='coerce')", language='python')
    
    st.markdown("กรองข้อมูลในช่วง 5 ปี")
    code='''end_date = pd.to_datetime('today')
start_date = end_date - pd.DateOffset(years=5)
data_filtered = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
'''
    st.code(code, language='python')

    st.markdown("การเติมค่าที่หายไป (Missing Values)")
    code = '''imputer = SimpleImputer(strategy='mean')
data_imputed = data_filtered.copy()
data_imputed['AQI'] = imputer.fit_transform(data_filtered[['AQI']])
'''
    st.code(code, language='python')

    st.markdown("แยกข้อมูล")
    code='''iX = data_imputed.drop(['Date', 'AQI'], axis=1)
y = data_imputed['AQI']
'''
    st.code(code, language='python')

    st.markdown("การเติมค่า NaN ใน X")
    st.code("X = imputer.fit_transform(X)", language='python')


    st.markdown("การปรับขนาดข้อมูล (Standardization)")
    code='''scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
'''
    st.code(code, language='python')

    st.markdown("การแบ่งข้อมูล (Scaling):")
    st.code("X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)", language='python')



    st.markdown("#### ทฤษฎีของอัลกอริทึมที่พัฒนา")

    st.markdown("1. Decision Tree Regressor : เป็นโมเดลที่สามารถตีความได้ง่าย ซึ่งหมายความว่าเราสามารถเห็นได้ชัดว่าโมเดลตัดสินใจอย่างไร")
    st.markdown("2. K-Nearest Neighbors (KNN) : เป็นอัลกอริธึมที่ไม่ต้องการการฝึกสอนล่วงหน้า ,  ไม่จำเป็นต้องปรับพารามิเตอร์ที่ซับซ้อนและง่ายต่อการทดสอบ , สามารถจัดการกับข้อมูลที่ไม่มีความสัมพันธ์เชิงเส้นได้ดี")
    st.markdown("3. Support Vector Regressor (SVR): : สำหรับการทำนายค่าต่อเนื่องในข้อมูลที่มีลักษณะเป็นเชิงเส้นและไม่เชิงเส้น ")
#3
    st.markdown("#### ขั้นตอนการพัฒนาโมเดล")
    st.markdown("ฝึกและทำนาย:")
    code='''dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)'''
    st.code(code, language='python')

    st.markdown(" การประเมินผลโมเดล")
    code='''mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
'''
    st.code(code, language='python')

    st.markdown("การแสดงกราฟ")
    code='''fig, axs = plt.subplots(3, 1, figsize=(10, 18))
axs[0].plot(y_test.values, label='Actual', color='blue', linestyle='-', linewidth=2)
axs[0].plot(y_pred_dt, label='Predicted', color='red', linestyle='--', linewidth=2)
'''
    st.code(code, language='python')

    





   



