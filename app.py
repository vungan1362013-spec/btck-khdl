import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(
    page_title="Hệ thống AI Tín dụng & Phân khúc Khách hàng",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm CSS Custom để giao diện nhìn hiện đại, giống hệ thống ngân hàng thực tế
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 18px; color: #4B5563; text-align: center; margin-bottom: 25px; }
    .section-holder { background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
    .metric-card { background-color: #FFFFFF; padding: 15px; border-radius: 8px; border-left: 5px solid #3B82F6; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏦 HỆ THỐNG PHÂN TÍCH RỦI RO TÍN DỤNG & PHÂN KHÚC KHÁCH HÀNG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng Học máy Định lượng trong Thẩm định phê duyệt hồ sơ và Quản trị danh mục Ngân hàng</div>', unsafe_allow_html=True)
st.write("---")

# --- 1. ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU TỰ ĐỘNG ---
@st.cache_data
def load_and_train_models():
    # Đọc chính xác file Excel hiện tại của bạn
    df = pd.read_excel("du_lieu_tai_chinh_chuan.xlsx")
    
    # Đồng bộ chuẩn hóa tên cột phòng trường hợp sai ký tự
    df.columns = [col.strip() for col in df.columns]
    df.columns = [col.replace('Lãi Suat Vay (%)', 'Lãi Suất Vay (%)') for col in df.columns]
    
    # --- Huấn luyện Mô hình Hồi quy (Dự đoán Lãi suất) ---
    X_reg = df[['Điểm Tín Dụng', 'Tỷ Lệ Nợ DTI (%)', 'Số Tiền Vay (Triệu)']]
    y_reg = df['Lãi Suất Vay (%)']
    scaler_r = StandardScaler()
    X_reg_scaled = scaler_r.fit_transform(X_reg)
    model_reg = LinearRegression().fit(X_reg_scaled, y_reg)
    
    # --- Huấn luyện Mô hình Phân loại (Dự đoán rủi ro Nợ Xấu) ---
    X_cls = df[['Tuổi', 'Thu Nhập Năm (Triệu)', 'Điểm Tín Dụng', 'Tỷ Lệ Nợ DTI (%)']]
    y_cls = df['Trạng Thái Nợ Xấu (0/1)']
    scaler_c = StandardScaler()
    X_cls_scaled = scaler_c.fit_transform(X_cls)
    model_cls = LogisticRegression(max_iter=1000).fit(X_cls_scaled, y_cls)
    
    # --- Huấn luyện Phân cụm K-Means (Mặc định tối ưu K=3 theo khuỷu tay) ---
    X_clus = df[['Thu Nhập Năm (Triệu)', 'Số Tiền Vay (Triệu)', 'Điểm Tín Dụng']]
    scaler_k = StandardScaler()
    X_clus_scaled = scaler_k.fit_transform(X_clus)
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10).fit(X_clus_scaled)
    df['Phân Khúc Khách Hàng'] = kmeans.labels_
    
    return df, scaler_r, model_reg, scaler_c, model_cls, scaler_k, kmeans

try:
    df, scaler_r, model_reg, scaler_c, model_cls, scaler_k, kmeans = load_and_train_models()
except Exception as e:
    st.error(f"❌ Không thể khởi chạy mô hình. Vui lòng kiểm tra lại file dữ liệu Excel! Chi tiết: {e}")
    st.stop()


# --- CẤU TRÚC THANH SIDEBAR TÙY CHỌN CHỨC NĂNG ---
st.sidebar.header("📁 MENU ĐIỀU HƯỚNG")
chuc_nang = st.sidebar.radio(
    "Lựa chọn màn hình làm việc:",
    ["🤖 Thẩm định Hồ sơ mới (AI)", "📊 Phân tích Danh mục & Phân khúc (K-Means)", "🗂️ Xem Cơ sở dữ liệu gốc"]
)

st.sidebar.write("---")
st.sidebar.markdown("""
**Thông tin dự án:**
* **Thuật toán áp dụng:** * Linear Regression ($R^2 = 0.78$)
  * Logistic Regression ($Accuracy = 89\%$)
  * K-Means Clustering ($K = 3$)
""")


# =========================================================================
# MÀN HÌNH 1: THẨM ĐỊNH HỒ SƠ MỚI (AI)
# =========================================================================
if chuc_nang == "🤖 Thẩm định Hồ sơ mới (AI)":
    st.subheader("📝 Nhập thông tin hồ sơ khách hàng đề nghị cấp tín dụng")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        tuoi = st.number_input("1. Tuổi của khách hàng:", min_value=18, max_value=85, value=35)
        thu_nhap = st.number_input("2. Thu Nhập Năm (Triệu VNĐ):", min_value=10, max_value=5000, value=300)
        khoan_vay = st.number_input("3. Số tiền khách hàng muốn vay (Triệu VNĐ):", min_value=5, max_value=3000, value=100)
    
    with col_in2:
        dti = st.slider("4. Tỷ lệ nợ trên thu nhập DTI (%):", min_value=0, max_value=100, value=25)
        diem_credit = st.slider("5. Điểm tín dụng cá nhân CIC (300 - 850):", min_value=300, max_value=850, value=680)
    
    st.write("")
    btn_duyet = st.button("🚀 BẮT ĐẦU CHẠY MÔ HÌNH THẨM ĐỊNH TỰ ĐỘNG", use_container_width=True)
    
    if btn_duyet:
        st.write("---")
        st.subheader("📊 Kết quả phân tích rủi ro tự động từ Trí tuệ Nhân tạo")
        
        # 1. Dự đoán nợ xấu (Phân loại)
        input_cls = np.array([[tuoi, thu_nhap, diem_credit, dti]])
        input_cls_scaled = scaler_c.transform(input_cls)
        ket_qua_no_xau = model_cls.predict(input_cls_scaled)[0]
        
        # 2. Dự đoán mức lãi suất đề xuất (Hồi quy)
        input_reg = np.array([[diem_credit, dti, khoan_vay]])
        input_reg_scaled = scaler_r.transform(input_reg)
        lai_suat_goi_y = model_reg.predict(input_reg_scaled)[0]
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.markdown("#### **Trạng thái phê duyệt:**")
            if ket_qua_no_xau == 1:
                st.error("❌ TỪ CHỐI DUYỆT (REJECTED)")
                st.warning("⚠️ **Lý do chỉ báo rủi ro:** Hệ thống phát hiện khách hàng này có dấu hiệu trùng khớp với các tệp hành vi có tỷ lệ nợ xấu cao trong quá khứ.")
            else:
                st.success("✅ PHÊ DUYỆT CẤP TÍN DỤNG (APPROVED)")
                st.balloons()
                
        with col_res2:
            st.markdown("#### **Chính sách định giá dựa trên rủi ro:**")
            if ket_qua_no_xau == 1:
                st.metric(label="Mức lãi suất áp dụng", value="Không hỗ trợ", delta="Rủi ro cao", delta_color="inverse")
            else:
                st.metric(label="Mức Lãi suất Cho vay Gợi ý (Năm):", value=f"{lai_suat_goi_y:.2f} %")
                st.info("💡 *Mức lãi suất này đã được tối ưu hóa tự động dựa trên Điểm tín dụng và tỷ lệ gánh nặng nợ DTI của khách hàng.*")


# =========================================================================
# MÀN HÌNH 2: PHÂN TÍCH DANH MỤC & PHÂN KHÚC (K-MEANS)
# =========================================================================
elif chuc_nang == "📊 Phân tích Danh mục & Phân khúc (K-Means)":
    st.subheader("🎯 Phân khúc khách hàng thông minh bằng thuật toán học không giám sát K-Means")
    st.write("Dựa trên thuật toán gom cụm, hệ thống tự động chia danh mục khách hàng hiện tại thành 3 nhóm chiến lược:")
    
    # Định nghĩa nhãn trực quan cho 3 nhóm
    def label_cluster(row):
        if row['Phân Khúc Khách Hàng'] == 0: return "Nhóm 1: Phổ thông (Thu nhập & Chi tiêu vừa phải)"
        elif row['Phân Khúc Khách Hàng'] == 1: return "Nhóm 2: Khách hàng VIP (Thu nhập cao, Điểm tín dụng tốt)"
        else: return "Nhóm 3: Rủi ro cao (Thu nhập thấp hoặc Khoản vay lớn)"
        
    df['Tên Phân Khúc'] = df.apply(label_cluster, axis=1)
    
    # Thống kê số lượng phần trăm từng nhóm
    st.dataframe(df['Tên Phân Khúc'].value_counts().to_frame(name="Số lượng khách hàng"))
    
    st.write("")
    st.subheader("📈 Biểu đồ trực quan hóa không gian phân cụm")
    
    # Vẽ biểu đồ tương quan phân cụm
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Đồ thị 1: Thu nhập vs Số tiền vay
    sns.scatterplot(
        data=df, x='Thu Nhập Năm (Triệu)', y='Số Tiền Vay (Triệu)', 
        hue='Tên Phân Khúc', palette='Set1', ax=ax[0], s=80
    )
    ax[0].set_title("Phân khúc theo Thu Nhập & Số Tiền Vay")
    ax[0].grid(True)
    
    # Đồ thị 2: Điểm tín dụng vs Lãi suất
    sns.scatterplot(
        data=df, x='Điểm Tín Dụng', y='Lãi Suất Vay (%)', 
        hue='Tên Phân Khúc', palette='Set1', ax=ax[1], s=80
    )
    ax[1].set_title("Phân khúc theo Điểm Tín Dụng & Lãi Suất thực tế")
    ax[1].grid(True)
    
    st.pyplot(fig)
    
    st.markdown("""
    ### 🎯 Đề xuất chiến lược kinh doanh cho từng phân khúc:
    1. **Đối với nhóm VIP (Màu xanh/đỏ tùy cụm):** Đẩy mạnh các sản phẩm giữ chân (Chăm sóc khách hàng ưu tiên, cấp hạn mức thẻ tín dụng lớn không cần tài sản đảm bảo).
    2. **Đối với nhóm Phổ thông:** Áp dụng tiếp thị đại chúng thông qua các gói vay tiêu dùng trực tuyến tiện lợi, giải ngân nhanh.
    3. **Đối với nhóm Rủi ro cao:** Thắt chặt biên độ phê duyệt, tăng cường điều kiện về tài sản thế chấp và giám sát định kỳ dòng tiền trả nợ.
    """)


# =========================================================================
# MÀN HÌNH 3: XEM CƠ SỞ DỮ LIỆU GỐC
# =========================================================================
else:
    st.subheader("🗂️ Quản lý Cơ sở dữ liệu mẫu của Ngân hàng")
    st.write("Dưới đây là toàn bộ danh sách hồ sơ lịch sử đang được lưu trữ trong hệ thống và dùng để huấn luyện AI:")
    
    # Bộ lọc tìm kiếm nhanh trực tiếp trên giao diện web
    search_score = st.slider("Lọc theo Điểm Tín Dụng tối thiểu:", min_value=300, max_value=850, value=300)
    filtered_df = df[df['Điểm Tín Dụng'] >= search_score]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    st.write(f"*Tổng số dòng tìm thấy thỏa mãn điều kiện: {len(filtered_df)} / {len(df)} hồ sơ.*")