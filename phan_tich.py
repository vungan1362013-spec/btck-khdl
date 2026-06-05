# thong ke va phan tich du lieu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, classification_report

# 1. ĐỌC FILE EXCEL ĐANG CÓ TRÊN MÁY
try:
    df = pd.read_excel("du_lieu_tai_chinh_chuan.xlsx")
    print("--- [OK] Đã tải bộ dữ liệu Excel thành công ---")
except Exception as e:
    print(f"Lỗi không đọc được file: {e}")

# Kịch bản đồng bộ tên cột khớp chính xác 100% với file Excel thực tế của bạn
X_reg_cols = ['Điểm Tín Dụng', 'Tỷ Lệ Nợ DTI (%)', 'Số Tiền Vay (Triệu)']
y_reg_col = 'Lãi Suat Vay (%)' if 'Lãi Suat Vay (%)' in df.columns else 'Lãi Suất Vay (%)'

# =========================================================================
# THUẬT TOÁN 1: HỒI QUY TUYẾN TÍNH (Dự đoán Lãi suất Vay)
# =========================================================================
print("\n=== [1] THUẬT TOÁN HỒI QUY TUYẾN TÍNH ===")
try:
    X_reg = df[X_reg_cols]
    y_reg = df[y_reg_col]

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)

    model_reg = LinearRegression()
    model_reg.fit(X_train_r_scaled, y_train_r)
    y_pred_r = model_reg.predict(X_test_r_scaled)

    print(f"Độ chính xác R-squared (R2 Score): {r2_score(y_test_r, y_pred_r):.4f}")
except Exception as e:
    print(f"Lỗi ở Thuật toán 1: {e}")


# =========================================================================
# THUẬT TOÁN 2: PHÂN LOẠI HỌC MÁY (Dự đoán Nợ Xấu)
# =========================================================================
print("\n=== [2] THUẬT TOÁN PHÂN LOẠI (LOGISTIC REGRESSION) ===")
try:
    X_cls = df[['Tuổi', 'Thu Nhập Năm (Triệu)', 'Điểm Tín Dụng', 'Tỷ Lệ Nợ DTI (%)']]
    y_cls = df['Trạng Thái Nợ Xấu (0/1)']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)

    model_cls = LogisticRegression(max_iter=1000)
    model_cls.fit(X_train_c_scaled, y_train_c)
    y_pred_c = model_cls.predict(X_test_c_scaled)

    print("Báo cáo kết quả phân loại mô hình tín dụng:")
    print(classification_report(y_test_c, y_pred_c))
except Exception as e:
    print(f"Lỗi ở Thuật toán 2: {e}")


# =========================================================================
# THUẬT TOÁN 3: GOM CỤM K-MEANS & VẼ BIỂU ĐỒ ELBOW
# =========================================================================
print("\n=== [3] THUẬT TOÁN GOM CỤM K-MEANS ===")
try:
    X_clus = df[['Thu Nhập Năm (Triệu)', 'Số Tiền Vay (Triệu)', 'Điểm Tín Dụng']]

    scaler_k = StandardScaler()
    X_clus_scaled = scaler_k.fit_transform(X_clus)

    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_clus_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='g')
    plt.title('Bieu do Khuu tay (Elbow Method)')
    plt.xlabel('So luong cum (K)')
    plt.ylabel('WCSS')
    plt.grid(True)

    plt.savefig("bieu_do_elbow.png")
    print("-> Đã chạy xong K-Means và lưu biểu đồ thành công tại 'bieu_do_elbow.png'")
except Exception as e:
    print(f"Lỗi ở Thuật toán 3: {e}")
