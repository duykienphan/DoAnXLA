# pip install tensorflow==2.15.0

from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Đường dẫn dữ liệu
train_path = "D:\\KienPhan\\Projects\\money_detection_rpi4\\dataset\\train"
valid_path = "D:\\KienPhan\\Projects\\money_detection_rpi4\\dataset\\valid"

# Data Generators
train_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    train_path, batch_size=16, class_mode='categorical'
)
valid_generator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
    valid_path, batch_size=16, class_mode='categorical'
)

# Xây dựng mô hình
base_model = MobileNetV2(weights="imagenet", include_top=False)

# Thêm các layer tùy chỉnh
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
print(model.summary())

# Làm đông toàn bộ các lớp pretrained
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)

# Early Stopping và Checkpoints
early_stop = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)
checkpoint = ModelCheckpoint(
    "money_model_best.h5", monitor="val_accuracy", save_best_only=True, verbose=1
)

# Huấn luyện mô hình
epochs = 20
model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    callbacks=[early_stop, checkpoint]
)

# Lưu mô hình sau huấn luyện
path_for_saved_model = "money_final_model.h5"
model.save(path_for_saved_model)