import os
import json
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, request, render_template
from PIL import Image

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# --- CẤU HÌNH ĐƯỜNG DẪN ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, 'model.tflite') # Sử dụng mô hình .tflite
CLASS_NAMES_PATH = os.path.join(APP_ROOT, 'class_names.json')
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- TẢI MÔ HÌNH TFLITE ---
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    print(">>> Mô hình TFLite và tên lớp đã được tải thành công!")
except Exception as e:
    print(f"!!! Lỗi khi tải mô hình TFLite: {e}")
    interpreter = None
    class_names = []

# --- HÀM XỬ LÝ ẢNH ---
def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# --- CÁC ROUTE CỦA WEB ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return render_template('index.html', error="Lỗi: Mô hình AI chưa được tải.")
    if 'file' not in request.files or not request.files['file'].filename:
        return render_template('index.html', error="Lỗi: Vui lòng chọn một tệp để tải lên.")

    file = request.files['file']
    try:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_image = preprocess_image(filepath)

        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index].replace('___', ' - ').replace('_', ' ')
        confidence = f"{np.max(prediction) * 100:.2f}%"
        final_prediction = f"{predicted_class_name} (Độ tin cậy: {confidence})"

        return render_template('index.html',
                               prediction=final_prediction,
                               image_file=filename)
    except Exception as e:
        app.logger.error(f"Lỗi xử lý: {e}")
        return render_template('index.html', error="Đã có lỗi xảy ra trong quá trình xử lý ảnh.")

if __name__ == '__main__':
    # Lấy cổng từ biến môi trường của Hugging Face, nếu không có thì mặc định là 7860
    port = int(os.environ.get('PORT', 7860))
    # Chạy ứng dụng với cổng được chỉ định
    app.run(host='0.0.0.0', port=port, debug=False)