# 📊 Data Mining App (Streamlit)

Ứng dụng khai phá dữ liệu trực quan bằng Streamlit, tự cài 3 thuật toán chính: **Apriori**, **Naive Bayes**, **Tập thô (Rough Set)**

---

## 🧱 Cấu trúc thư mục

```
data-mining-app/
├── ui/                      # Giao diện chính (app.py)
├── module/                 # Thuật toán tự cài đặt
│   ├── apriori_custom.py       # Luật kết hợp (không tăng cường)
│   ├── naive_bayes_custom.py   # Naive Bayes
│   └── rough_set_custom.py     # Tập thô
├── data/                   # Dữ liệu mẫu (tùy thêm)
├── requirements.txt        # Thư viện cần cài
├── README.md               # Hướng dẫn sử dụng
```

---

## 🚀 Cách chạy ứng dụng

### 1. Clone về máy:
```bash
git clone https://github.com/khoicahu204/data-mining-app.git
cd data-mining-app
```

### 2. Cài thư viện:
```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng:
```bash
streamlit run ui/app.py
```

👉 Trình duyệt sẽ mở tại `http://localhost:8501`

---

## 🧠 Các mô hình hỗ trợ

### 📊 Apriori (Luật kết hợp)
- Người dùng upload file CSV
- Chọn cột giao dịch & mặt hàng
- Chọn thuật toán Apriori / Không tăng cường
- Nhập min_support & confidence
- Hiển thị tập phổ biến, luật kết hợp
- Tải kết quả CSV + biểu đồ

---

### 🧠 Naive Bayes (Phân lớp)
- Chọn cột mục tiêu & thuộc tính
- Nhập giá trị thuộc tính để dự đoán
- Dự đoán lớp với độ chính xác 100%
- Hiển thị log-xác suất từng lớp

---

### 📘 Tập thô (Rough Set)
- Chọn cột điều kiện & quyết định
- Chọn chức năng:
  - Xấp xỉ dưới/trên
  - Mức độ phụ thuộc
  - Tìm reduct
  - Sinh luật chính xác 100%
- Tải luật về CSV nếu cần

---

## 📩 Đóng góp / liên hệ

Bạn muốn mở rộng thêm mô hình? Giao diện? Tích hợp dữ liệu mới?

> Liên hệ: `khoicahu204@gmail.com`