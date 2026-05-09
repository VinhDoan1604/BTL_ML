# Phân loại Đánh giá Khách hàng trên Shopee (Shopee Customer Review Classification)

## MỤC TIÊU CỦA BÀI TẬP LỚN
Mục tiêu của dự án là áp dụng các kỹ thuật Học máy (Machine Learning) và Học sâu (Deep Learning) để xây dựng mô hình phân loại tự động mức độ đánh giá (từ 1 đến 5 sao) của khách hàng dựa trên nội dung văn bản (text) thu thập từ nền tảng thương mại điện tử Shopee. Dự án nhằm rèn luyện và ứng dụng kỹ năng thu thập, tiền xử lý dữ liệu ngôn ngữ tự nhiên (NLP), trích xuất đặc trưng (embedding) và tối ưu hóa các thuật toán phân loại.

---

## THÔNG TIN CHUNG
* **Tên môn học:** Học máy
* **Mã môn học:** CO3117
* **Học kỳ:** HK252
* **Năm học:** 2025 - 2026
* **Giảng viên hướng dẫn (GVHD):** TS. Trương Vĩnh Lân

### Thông tin các thành viên nhóm
| STT | Họ và Tên | MSSV |
|:---:|:---|:---:|
| 1 | Nguyễn Nhật Thiên Hữu| 2311382 |
| 2 | Nguyễn Lê Thảo Ly | 2312010 |
| 3 | Đoàn Công Vinh | 2313906 |
| 4 | Huỳnh Duy Chương | 2310363 |

---

## LIÊN KẾT DỰ ÁN & DỮ LIỆU
* **Báo cáo chi tiết (PDF):** [Link Google Drive / OneDrive đính kèm sau]
* **Colab Notebook:** [Colab](https://colab.research.google.com/github/VinhDoan1604/BTL_ML/blob/main/main.ipynb)
* **Dữ liệu đặc trưng (Embedding files):** [Files](https://drive.google.com/drive/folders/1KX2kw95f19eg2auDQTnq1WrZxtJgT16k?usp=sharing)
* **Dataset:** [Shopee Reviews Dataset](https://www.kaggle.com/datasets/shymammoth/shopee-reviews)

---

## HƯỚNG DẪN CHẠY DỰ ÁN

Dự án được thiết kế để chạy trực tiếp trên nền tảng **Google Colab** nhằm tận dụng tài nguyên GPU miễn phí.

1.  **Truy cập Notebook**
2.  **Thiết lập môi trường:**
    * Vào menu `Runtime` -> `Change runtime type`.
    * Chọn **Hardware accelerator** là **T4 GPU**.
3.  **Thực thi:**
    * Sử dụng menu `Runtime` -> `Run all` để chạy toàn bộ quy trình từ tải dữ liệu đến huấn luyện.

---

## QUY TRÌNH XỬ LÝ (PIPELINE)

Hệ thống được xây dựng theo một pipeline khép kín bao gồm:

1.  **Khám phá dữ liệu (EDA):** Phân tích phân phối nhãn, trực quan hóa các đặc trưng định lượng như số lượng từ, tỷ lệ viết hoa và biểu tượng cảm xúc.
2.  **Tiền xử lý văn bản (Preprocessing):**
    * **Normalization:** Chuyển về chữ thường, xử lý ký tự lặp và các ký tự đặc biệt.
    * **Emoji Handling:** Chuyển đổi biểu tượng cảm xúc thành văn bản tương ứng và khử nhiễu lặp.
    * **Tokenization & Lemmatization:** Sử dụng NLTK và WordNet để đưa từ về dạng gốc theo ngữ cảnh (POS Tagging).
3.  **Trích xuất đặc trưng (Embedding):**
    * Áp dụng TF-IDF và BoW trực tiếp cho luồng văn bản đã qua xử lí. 
    * Sử dụng SentenceTransformer để trích xuất ngữ nghĩa văn bản, sau đó lưu toàn bộ vector dưới định dạng .npy.
4.  **Mô hình hóa (Modeling):**
    * **Machine Learning:** Logistic Regression, Naive Bayes.
    * **Deep Learning:** LSTM, RNN.
    * **Transformers:** BERT/DistilBERT (Pre-trained models).
5.  **Đánh giá:** Đo lường bằng Accuracy, F1-Score và Confusion Matrix.

---

## CẤU TRÚC THƯ MỤC
```text
BTL_ML/
├── notebooks/
├── modules/
│   ├── preprocessing.py   # Các hàm làm sạch, tokenization, lemmatization
│   ├── models.py          # Định nghĩa kiến trúc các mô hình
│   ├── eda.py             # Các hàm bổ trợ phân tích và trực quan hoá dữ liệu
│   └── cleaning.py        # Các hàm xử lý nhãn và loại bỏ nhiễu ban đầu
├── reports/              # Chứa file báo cáo PDF hoàn chỉnh và các tài liệu thuyết trình
├── features/             # Vector đặc trưng/ embedding đã trích xuất (.npy)
├── requirements.txt      # Danh sách thư viện cần thiết
└── README.md           
