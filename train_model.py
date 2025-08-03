import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import json
import os

# 1. Cấu hình các tham số
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = 'dataset'

# 2. Chuẩn bị dữ liệu
# Sử dụng ImageDataGenerator để tải và tăng cường dữ liệu (data augmentation)
# Rescale để chuẩn hóa giá trị pixel về khoảng [0, 1]
# Validation_split để tự động chia tập dữ liệu thành training và validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # 20% dữ liệu dùng để kiểm tra
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Tạo dữ liệu huấn luyện
train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training' # Chỉ định đây là tập huấn luyện
)

# Tạo dữ liệu kiểm tra
validation_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation' # Chỉ định đây là tập kiểm tra
)

# 3. Xây dựng mô hình (Sử dụng Transfer Learning với MobileNetV2)
# Tải mô hình MobileNetV2 đã được huấn luyện trước trên bộ dữ liệu ImageNet
# include_top=False để bỏ lớp phân loại cuối cùng, vì ta sẽ thay bằng lớp của mình
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# "Đóng băng" các lớp của base_model để không huấn luyện lại chúng
base_model.trainable = False

# Xây dựng các lớp phân loại mới
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# Lớp output có số nơ-ron bằng số lớp (số loại bệnh)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Bắt đầu quá trình huấn luyện...")

# 5. Huấn luyện mô hình
# Quá trình này có thể mất thời gian tùy thuộc vào cấu hình máy tính của bạn
history = model.fit(
    train_generator,
    epochs=10, # Chạy 10 lần qua toàn bộ dữ liệu. Tăng lên nếu muốn độ chính xác cao hơn.
    validation_data=validation_generator
)

# 6. Lưu mô hình và tên các lớp
print("Huấn luyện hoàn tất! Đang lưu mô hình...")
model.save('plant_disease_model.h5')

# Lưu tên các lớp (tên các thư mục bệnh) vào một file json
class_names = list(train_generator.class_indices.keys())
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

print("Đã lưu mô hình vào 'plant_disease_model.h5'")
print("Đã lưu tên các lớp vào 'class_names.json'")