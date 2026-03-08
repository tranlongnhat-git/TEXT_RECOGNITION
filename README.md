# 🚀 TEXT_RECOGNITION (OCR System)

Hệ thống nhận diện chữ viết (Optical Character Recognition - OCR) sử dụng mô hình học sâu **CRNN** (Convolutional Recurrent Neural Network) kết hợp với hàm mất mát **CTC Loss**. Project đi kèm với một giao diện đồ họa (GUI) thân thiện giúp người dùng dễ dàng chuyển đổi hình ảnh chứa văn bản thành văn bản số.

## 📋 Tính năng chính
* **Nhận diện đa dạng:** Hỗ trợ nhận diện chữ cái tiếng Anh (a-z, A-Z) và chữ số (0-9).
* **Kiến trúc mạnh mẽ:** Sử dụng CNN để trích xuất đặc trưng và Bi-LSTM để hiểu ngữ cảnh chuỗi ký tự.
* **Giao diện trực quan:** GUI xây dựng bằng Tkinter, hỗ trợ chọn ảnh, xem trước và lưu kết quả ra file `output.txt`.
* **Tiền xử lý thông minh:** Tự động chuẩn hóa kích thước ảnh về chuẩn 128x32 pixel để đạt độ chính xác cao nhất.

---

## 🏗 Kiến trúc mô hình
Mô hình được xây dựng theo cấu trúc mạng nơ-ron tích chập kết hợp chuỗi (CRNN):



1.  **CNN (Feature Extraction):** 7 lớp Convolutional với số bộ lọc tăng dần từ 64 đến 512, giúp trích xuất các đặc trưng hình ảnh của ký tự.
2.  **Squeeze & Reshape:** Chuyển đổi dữ liệu từ dạng bản đồ đặc trưng (feature map) sang dạng chuỗi thời gian.
3.  **RNN (Sequence Labeling):** 2 lớp **Bidirectional LSTM** (128 units) giúp nắm bắt thông tin theo cả hai chiều của văn bản.
4.  **CTC Loss:** Hàm mất mát Connectionist Temporal Classification (CTC) giúp huấn luyện mô hình mà không cần gán nhãn cho từng pixel vị trí ký tự cụ thể.

---

## 📂 Cấu trúc thư mục
* `model.py`: Định nghĩa cấu trúc mạng CNN + Bi-LSTM và hàm CTC.
* `predict.py`: File chạy chính (Main GUI). Chứa code xử lý giao diện Tkinter và logic dự đoán.
* `train.py`: Script dùng để huấn luyện mô hình.
* `best_model3.keras` / `best_model5.keras`: Các file trọng số mô hình đã được huấn luyện.

---

## 💾 Dữ liệu huấn luyện (Dataset)

Dự án này được thiết lập để sử dụng tập dữ liệu **MJSynth (90k Dictionary)** hoặc các tập dữ liệu tự tạo (Custom Dataset) tuân thủ đúng định dạng.

3. Cấu hình đường dẫn
Đảm bảo bạn đã trỏ đúng đường dẫn tuyệt đối đến thư mục chứa dữ liệu trong file train.py:

Python
path = r'D:\Download\90kDICT32px'
4. Tiền xử lý ảnh (Tự động)
Khi huấn luyện, mô hình sẽ tự động xử lý:

Chuyển ảnh màu (RGB) sang ảnh xám (Grayscale).

Bỏ qua các ảnh có kích thước lớn hơn 128x32.

Tự động đệm (padding) pixel trắng (giá trị 255) cho các ảnh nhỏ hơn để vừa với đầu vào 128x32.

🛠 Cài đặt & Sử dụng
1. Yêu cầu hệ thống
Cài đặt môi trường Python (khuyên dùng Python 3.8+) và các thư viện cần thiết:

Bash
pip install tensorflow numpy opencv-python pillow
2. Tải trọng số mô hình (Pre-trained Model)
Nếu bạn chỉ muốn chạy thử giao diện nhận diện mà không cần train lại, hãy tải file mô hình đã huấn luyện sẵn và đặt cùng thư mục với file code:

Link tải model: Download file weights(https://drive.google.com/file/d/19cuiRz4Cuvn13ozmrEf553LMnejZPZ3R/view?usp=drive_link)
(Lưu ý: Bạn cần đổi tên file tải về thành best_model3.keras để code trong predict.py load đúng tên).

3. Chạy ứng dụng (GUI)
Để mở giao diện phần mềm:

Bash
python predict.py
Hướng dẫn sử dụng:

Nhấn "Chọn ảnh" để tải hình ảnh văn bản lên.

Nhấn "Chuyển đổi" để mô hình thực hiện nhận diện.

Nhấn "Thêm vào file" để lưu kết quả văn bản vào file output.txt.

4. Huấn luyện lại mô hình (Training)
Nếu bạn có tập dữ liệu mới và muốn tự train:

Bash
python train.py
📊 Thông số huấn luyện
Input Size: 128x32 (Grayscale)

Vocabulary: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789

Optimizer: Adam

Batch Size: 256

Epochs: 20

Callbacks: Lưu lại mô hình có val_loss thấp nhất.
