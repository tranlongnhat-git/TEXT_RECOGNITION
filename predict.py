import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from model import create_model
import string

char_list = string.ascii_letters + string.digits

def decode_predict_ctc(out, char_list):
    out_best = np.argmax(out[0], axis=1)
    out_text = ''
    for c in out_best:
        if c < len(char_list):
            out_text += char_list[c]
    return out_text

def preprocess_image(image):

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    scale = min(128 / w, 32 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img_resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_padded = np.full((32, 128), 255, dtype=np.uint8)
    img_padded[:new_h, :new_w] = img_resized

    img_normalized = np.expand_dims(img_padded, axis=2) / 255.0
    return np.expand_dims(img_normalized, axis=0)

_, act_model = create_model()
act_model.load_weights("best_model3.keras")

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện chữ in")

        self.root.geometry("800x400")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.image_label = tk.Label(self.main_frame, bg="gray")
        self.image_label.place(relx=0.5, rely=0.5, anchor="center")


        self.result_text = tk.Text(
            self.main_frame,
            height=4,
            font=("Arial", 20),
            fg="blue",
            wrap=tk.WORD
        )
        self.result_text.pack(pady=10)

        self.info_label = tk.Label(
            self.main_frame,
            text="Hướng dẫn: Chọn ảnh rõ nét, không mờ, kích thước tối thiểu 128x32.",
            font=("Arial", 10),
            fg="gray"
        )
        self.info_label.pack(pady=5)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        self.choose_button = tk.Button(
            self.button_frame,
            text="Chọn ảnh",
            command=self.load_image,
            width=15,
            bg="lightblue"
        )
        self.choose_button.pack(side=tk.LEFT, padx=20)

        self.predict_button = tk.Button(
            self.button_frame,
            text="Chuyển đổi",
            command=self.predict_text,
            state=tk.DISABLED,
            width=15,
            bg="lightgreen"
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(
            self.button_frame,
            text="Thêm vào file",
            command=self.save_to_file,
            state=tk.DISABLED,
            width=15,
            bg="lightyellow"
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.image = None
        self.current_result = ""

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image = cv2.imread(file_path)

            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            max_display_size = (600, 400)
            image_pil.thumbnail(max_display_size, Image.Resampling.LANCZOS)

            image_tk = ImageTk.PhotoImage(image_pil)

            self.image_label.configure(image=image_tk)
            self.image_label.image = image_tk

            self.root.geometry(f"{max(image_pil.width + 20, 800)}x{max(image_pil.height + 100, 600)}")

            self.predict_button.config(state=tk.NORMAL)

    def predict_text(self):
        if self.image is None:
            messagebox.showerror("Lỗi", "Vui lòng chọn một ảnh!")
            return

        h, w, _ = self.image.shape
        if w < 20 or h < 20:
            messagebox.showerror("Lỗi", "Ảnh quá nhỏ để nhận diện!")
            return

        img_input = preprocess_image(self.image)

        pred = act_model.predict(img_input)
        pred_text = decode_predict_ctc(pred, char_list)


        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, pred_text)
        self.current_result = pred_text

        self.save_button.config(state=tk.NORMAL)

    def save_to_file(self):
        if not self.current_result:
            messagebox.showerror("Lỗi", "Không có kết quả để lưu!")
            return

        try:
            with open("output.txt", "a", encoding="utf-8") as file:
                file.write(self.current_result + "\n")
            messagebox.showinfo("Thành công", "Kết quả đã được lưu vào file output.txt")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu file: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()

