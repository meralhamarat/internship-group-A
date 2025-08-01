import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# --- MediaPipe Modelleri ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Parametreler ---
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
SEQUENCE_LENGTH = 15  # 30'dan 15'e düşürdük - daha hızlı tepki
FRAME_SKIP_RATE = 3   # 5'ten 3'e düşürdük - daha sık örnekleme

def load_trained_model():
    """Eğitilmiş modeli ve label encoder'ı yükle"""
    try:
        model = load_model('best_action_model.h5')
        label_encoder = joblib.load('label_encoder.pkl')
        print("Model ve label encoder başarıyla yüklendi!")
        print(f"Sınıflar: {label_encoder.classes_}")
        return model, label_encoder
    except Exception as e:
        print(f"Model yüklenirken hata: {e}")
        return None, None

def extract_pose_keypoints(pose_landmarks):
    """Pose landmark'larından keypoint'leri çıkar"""
    if pose_landmarks:
        keypoints = []
        for landmark in pose_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y])
        return keypoints
    else:
        # Pose algılanmadıysa sıfır dolu array döndür
        return [0.0] * 66  # 33 landmarks x 2 coordinates

def predict_action(model, label_encoder, keypoints_sequence):
    """Keypoint dizisinden hareket tahmini yap"""
    if len(keypoints_sequence) < SEQUENCE_LENGTH:
        return "Insufficient Data", 0.0
    
    # Son SEQUENCE_LENGTH karesini al
    recent_sequence = keypoints_sequence[-SEQUENCE_LENGTH:]
    
    # Numpy array'e çevir ve reshape et
    sequence_array = np.array(recent_sequence).reshape(1, SEQUENCE_LENGTH, -1)
    
    # Tahmin yap
    predictions = model.predict(sequence_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # Sınıf adını al
    predicted_action = label_encoder.inverse_transform([predicted_class_index])[0]
    
    return predicted_action, confidence

# ... (Diğer kodlar değişmeden kalıyor) ...

def main():
    # Modeli yükle
    model, label_encoder = load_trained_model()
    if model is None:
        print("Model yüklenemedi. Lütfen önce train_classifier.py çalıştırın.")
        return

    VIDEO_SOURCE = 'man.mp4'
    if isinstance(VIDEO_SOURCE, str) and not os.path.exists(VIDEO_SOURCE):
        print(f"Uyarı: '{VIDEO_SOURCE}' dosyası bulunamadı!")
        print("Lütfen test videonuzu bu klasöre koyun ve dosya adını kontrol edin.")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Video kaynağı açılamadı: {VIDEO_SOURCE}")
        return

    keypoints_history = deque(maxlen=SEQUENCE_LENGTH * 2)
    frame_count = 0
    resized_window = False

    print("Real-time hareket tanıma başlatıldı...")
    print("Çıkış için ESC tuşuna basın")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % FRAME_SKIP_RATE != 0:
                continue

            if not resized_window:
                h, w, _ = frame.shape
                scale_w = MAX_DISPLAY_WIDTH / w
                scale_h = MAX_DISPLAY_HEIGHT / h
                scale = min(scale_w, scale_h, 1.0)
                display_w = int(w * scale)
                display_h = int(h * scale)
                cv2.namedWindow('Real-time Action Recognition', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Real-time Action Recognition', display_w, display_h)
                resized_window = True

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            keypoints = extract_pose_keypoints(results.pose_landmarks)
            keypoints_history.append(keypoints)

            if len(keypoints_history) >= SEQUENCE_LENGTH:
                predicted_action, confidence = predict_action(model, label_encoder, list(keypoints_history))

                # Güven seviyesine göre renk belirle (bu kısım değişmedi)
                if confidence > 0.7:
                    color = (0, 255, 0)
                elif confidence > 0.5:
                    color = (0, 255, 255)
                else:
                    color = (0, 0, 255)

                recent_sequence = list(keypoints_history)[-SEQUENCE_LENGTH:]
                sequence_array = np.array(recent_sequence).reshape(1, SEQUENCE_LENGTH, -1)
                predictions = model.predict(sequence_array, verbose=0)[0]

                # Tespiti her zaman yazdır (eski if bloğu kaldırıldı)
                cv2.putText(image, f'Action: {predicted_action}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

                cv2.putText(image, f'Confidence: {confidence:.2f}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                y_offset = 130
                for i, class_name in enumerate(label_encoder.classes_):
                    score = predictions[i]
                    text_color = (0, 255, 0) if i == np.argmax(predictions) else (255, 255, 255)
                    cv2.putText(image, f'{class_name}: {score:.3f}', (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2 if i == np.argmax(predictions) else 1, cv2.LINE_AA)
                    y_offset += 25
            else:
                cv2.putText(image, 'Collecting frames...', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, f'{len(keypoints_history)}/{SEQUENCE_LENGTH}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            cv2.putText(image, 'Press ESC to exit', (10, image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if resized_window:
                cv2.imshow('Real-time Action Recognition', cv2.resize(image, (display_w, display_h)))
            else:
                cv2.imshow('Real-time Action Recognition', image)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time hareket tanıma sonlandırıldı.")

if __name__ == "__main__":
    main()