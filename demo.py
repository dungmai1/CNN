from keras.models import load_model
from PIL import Image
import numpy as np
from tkinter import Tk, filedialog
from keras.preprocessing import image

#Load mô hình
model = load_model("catvsdog.h5", compile=False)

#Chọn hình ảnh sử dụng thử viện tkinter để hiển thị hộp thoại lựa chọn file
root = Tk()
root.withdraw()  
file_path = filedialog.askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
if file_path:
    image1 = Image.open(file_path).convert("RGB")
else:
    print("No file selected.")

# Chuyển đổi hình ảnh thành mảng và điều chỉnh kích thước
img_array = image.img_to_array(image1)
img_array = np.expand_dims(img_array, axis=0) 
img_array = image.smart_resize(img_array, (224, 224))

#Dự đoán bằng mô hình
prediction = model.predict(img_array)
threshold = 0.5

# So sánh với ngưỡng và in kết quả
if prediction[0][0] < threshold:
    print('cat')
else:
    print('dog')
