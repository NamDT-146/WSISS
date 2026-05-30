Dưới đây là tổng hợp thông tin chi tiết về hai mô hình **SAM (phiên bản Base)** và **Mask2Former (phiên bản Swin-T)** dựa trên các nguồn tài liệu và lịch sử trao đổi:

### **1. Segment Anything Model (SAM) - Phiên bản Base (ViT-B)**
*   **Kiến trúc:** Sử dụng bộ mã hóa hình ảnh (image encoder) là **Vision Transformer (ViT-B)** được tiền huấn luyện bằng phương pháp MAE [81, Turn 21].
*   **Kích thước hình ảnh đầu vào (Input Size):** Trong cấu hình chuẩn, hình ảnh được resize và thêm padding để đạt kích thước cố định **$1024 \times 1024$** pixel trước khi đưa vào encoder [Turn 20, Turn 24].
*   **Kích thước đầu ra của Encoder (Output Size):** Với đầu vào $1024 \times 1024$, bộ mã hóa tạo ra một bản đồ đặc trưng (image embedding) kích thước **$64 \times 64$** (tương ứng với stride 16) [81, Turn 21, Turn 24].
*   **Kích thước nhúng (Embedding Size):** Chiều của image embedding là **256** kênh [Turn 20, Turn 21].
*   **Kho mã nguồn (Code Repository):** Triển khai chính thức bằng PyTorch bởi Meta AI (FAIR) có tại địa chỉ: **`https://segment-anything.com`** (Giấy phép Apache 2.0).

### **2. Mask2Former - Phiên bản Swin-Tiny (Swin-T)**
*   **Kiến trúc:** Sử dụng bộ khung (backbone) là **Swin Transformer** phân cấp phiên bản **Tiny** [175, 202, Turn 26].
*   **Kích thước hình ảnh đầu vào (Input Size):** Kích thước thay đổi tùy theo tập dữ liệu:
    *   **ADE20K:** Thường sử dụng kích thước cắt (crop size) là **$512 \times 512$**.
    *   **COCO:** Huấn luyện với các bản cắt **$1024 \times 1024$** và suy luận với cạnh ngắn nhất là 800 pixel [26, Turn 20].
*   **Kích thước đầu ra của Encoder (Output Size):** Khác với SAM, Swin-T tạo ra tháp đặc trưng đa quy mô (multi-scale). Các mức độ phân giải chính (strides) bao gồm **8, 16, và 32** [17, Turn 27]. Ví dụ, với đầu vào $512 \times 512$, cấp độ sâu nhất (stride 32) sẽ có kích thước **$16 \times 16$** [Turn 27].
*   **Kích thước nhúng (Embedding Size):** Tất cả các đặc trưng từ backbone sau khi qua Pixel Decoder đều được chiếu về mức **256** kênh để đưa vào Transformer decoder [169, 171, Turn 27].
*   **Kho mã nguồn (Code Repository):** Dự án chính thức được duy trì tại: **`https://bowenc0221.github.io/mask2former`**. Mã nguồn này được xây dựng trên framework **Detectron2** (`https://github.com/facebookresearch/detectron2`).

### **Bảng so sánh tóm tắt**

| Đặc điểm | SAM (Base/ViT-B) | Mask2Former (Swin-T) |
| :--- | :--- | :--- |
| **Kiến trúc Encoder** | Vision Transformer (Phẳng) | Swin Transformer (Phân cấp) |
| **Input Size chuẩn** | **$1024 \times 1024$** [Turn 20] | **$512 \times 512$** hoặc **$1024 \times 1024$** |
| **Encoder Output** | Đơn quy mô (**$64 \times 64$**) [Turn 24] | Đa quy mô (**Strides 8, 16, 32**) [17, Turn 27] |
| **Embedding Dim** | **256** [Turn 21] | **256** (sau khi chiếu) |
| **Framework** | PyTorch | PyTorch (Detectron2) |