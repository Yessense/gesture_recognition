import os

# Работа с изображениями
import cv2
# Поиск точек
import mediapipe as mp
# Вычисления матриц
import numpy as np
# Сохранение и загрузка файлов
from joblib import dump, load

# Константы в программе

# Наши классы
LABELS = {
    0: 'stand',
    1: 'jump',
    2: 'place',
    3: 'down',
    4: 'sit',
    5: 'come',
    6: 'paw',
}

# Расстояния, с которых снят датасет
DISTANCES = {
    0: 'down_close',
    1: 'down_far',
    2: 'middle_close',
}

# название: число
LABELS_R = {value: key for key, value in LABELS.items()}
DISTANSES_R = {value: key for key, value in DISTANCES.items()}

# Пусть к папке с датасетом
DATASET_DIR = '/media/yessense/Transcend/gestures_dataset'

# Куда сохранять аннотированные изображения
ANNOTATED_DIR = '/media/yessense/Transcend/annotated_images'

# Настройка библиотеки mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


IMAGE_FILES = []

# Точки на изображении
x_train = []
# Класс изображения
y_train = []

# Считываем все картинки из папки
for gesture in os.listdir(DATASET_DIR):
    for file in os.listdir(os.path.join(DATASET_DIR, gesture)):
        IMAGE_FILES.append((gesture, os.path.join(DATASET_DIR, gesture), file))

BG_COLOR = (192, 192, 192)  # gray



with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5) as pose:
    # Итерируемся по каждому изображению в датасете
    for gesture, directory, file in IMAGE_FILES:
        # считываем изображение
        file_path = os.path.join(directory, file)
        image = cv2.imread(os.path.join(directory, file))
        image_height, image_width, _ = image.shape

        # TODO: Здесь можно аугментировать изображение

        images = augmented_images()

        for image in images:

            # Convert the BGR image to RGB before processing.
            # Запуск модели на картинку
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Если нет точек, то пропускаем изображение
            if not results.pose_landmarks:
                continue

            # Записываем все точки в один лист
            person = []
            for point in results.pose_landmarks.landmark:
                person.append([point.x,
                               point.y,
                               point.z,
                               point.visibility])

            # Записываем проработанное изображение в список
            x_train.append(person)
            y_train.append(LABELS_R[gesture])

            print(f'{file}')


            # Рисуем аннотации на изображении
            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            # bg_image = np.zeros(image.shape, dtype=np.uint8)
            # bg_image[:] = BG_COLOR
            # annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # cv2.imwrite(os.path.join(annotated_dir, gesture, file + '.png'), annotated_image)
            # Plot pose world landmarks.
            # mp_drawing.plot_landmarks(
            #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Сохраняем на диск наши тренировочные данные
    np.save('../data/data.npy', x_train)
    np.save('../data/labels.npy', y_train)
