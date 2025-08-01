# Bu betik, önceden eğitilmiş bir makine öğrenmesi modeli kullanmak yerine,
# insan vücudunun anlık geometrik ve zamansal özelliklerini (açı, mesafe, hız gibi) analiz ederek hareketleri tanımlar.
# Sistem, belirlediği eşik değerlere ve kurallara dayanarak Sitting, Standing, Clapping gibi hareketleri sınıflandırır.
# Bu yaklaşım, basit ve hızlı bir hareket tanıma sistemi oluşturmak için idealdir.
import cv2
import mediapipe as mp
import numpy as np
import os
import math
from collections import deque
import time

# --- MediaPipe Modelleri ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

# --- Parametreler ---
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
ACTION_HISTORY_BUFFER_SIZE = 20 # Hareketin kararlılığı için bakılacak kare sayısı
MIN_CONFIDENT_FRAMES = 8       # Bir hareketin kararlı sayılması için gereken minimum kare sayısı
LOG_FILE = 'action_log_simple.json' # Basitleştirilmiş log dosyası

# Global değişkenler (display ve hareket analizi için)
last_predicted_label = "Tanımlanıyor..."
current_action_start_time = time.time()
action_history_log = []

# --- Yardımcı Fonksiyonlar ---

def calculate_distance(lm1, lm2):
    """İki MediaPipe landmark'ı arasındaki mesafeyi hesaplar."""
    return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)

def calculate_angle(a_lm, b_lm, c_lm):
    """Üç MediaPipe landmark'ı arasındaki açıyı derece cinsinden hesaplar."""
    a = np.array([a_lm.x, a_lm.y])
    b = np.array([b_lm.x, b_lm.y])
    c = np.array([c_lm.x, c_lm.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360 - angle
    return angle

def analyze_pose(pose_landmarks, hands_landmarks):
    """
    Sadece 3 temel hareketi (Oturma, Ayakta Durma, Alkışlama) analiz eder.
    """
    current_action = "Unknown"
    
    # --- Alkışlama Tespiti ---
    if hands_landmarks and len(hands_landmarks) >= 2:
        wrist1 = hands_landmarks[0].landmark[0]
        wrist2 = hands_landmarks[1].landmark[0]
        
        distance_between_hands = calculate_distance(wrist1, wrist2)
        
        # Eller çok yakınsa alkışlama olarak kabul et
        if distance_between_hands < 0.08: # Bu eşik deneysel olarak ayarlanabilir
            return "Clapping"

    # --- Duruş (Vücut) Tespiti ---
    if not pose_landmarks:
        return current_action

    # Landmark isimleri
    LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP
    RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP
    LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE
    RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE
    LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER
    RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER

    # Kritik noktaların görünürlüğünü kontrol et
    min_visibility_threshold = 0.6
    critical_landmarks = [
        pose_landmarks.landmark[LEFT_HIP], pose_landmarks.landmark[RIGHT_HIP],
        pose_landmarks.landmark[LEFT_KNEE], pose_landmarks.landmark[RIGHT_KNEE],
        pose_landmarks.landmark[LEFT_SHOULDER], pose_landmarks.landmark[RIGHT_SHOULDER]
    ]
    if any(lm.visibility < min_visibility_threshold for lm in critical_landmarks):
        return current_action

    # Ortalama diz açısı
    left_leg_angle = calculate_angle(pose_landmarks.landmark[LEFT_HIP], pose_landmarks.landmark[LEFT_KNEE], pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE])
    right_leg_angle = calculate_angle(pose_landmarks.landmark[RIGHT_HIP], pose_landmarks.landmark[RIGHT_KNEE], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE])
    avg_leg_angle = (left_leg_angle + right_leg_angle) / 2

    # Omuz-kalça arasındaki dikey mesafe
    shoulder_y = (pose_landmarks.landmark[LEFT_SHOULDER].y + pose_landmarks.landmark[RIGHT_SHOULDER].y) / 2
    hip_y = (pose_landmarks.landmark[LEFT_HIP].y + pose_landmarks.landmark[RIGHT_HIP].y) / 2
    height_diff = abs(shoulder_y - hip_y)

    # --- Sınıflandırma Mantığı ---
    # Oturma: Vücut sıkışık ve dizler bükülü
    if height_diff < 0.25 and avg_leg_angle < 130:
        current_action = "Sitting"
    
    # Ayakta Durma: Bacaklar nispeten düz ve vücut uzun (eşikler daha esnek)
    elif avg_leg_angle > 140 and height_diff > 0.2:  # 160->140, 0.25->0.2 daha esnek
        current_action = "Standing"
        
    return current_action

def log_action_if_changed(current_label):
    """
    Hareket değiştiğinde bir önceki hareketi ve süresini loglar.
    """
    global last_predicted_label, current_action_start_time, action_history_log

    if current_label != last_predicted_label and last_predicted_label != "Tanımlanıyor...":
        duration = time.time() - current_action_start_time
        action_history_log.append({
            "action": last_predicted_label,
            "duration_seconds": round(duration, 2),
            "end_time": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        print(f"Log: '{last_predicted_label}' - Süre: {round(duration, 2)} saniye")
        current_action_start_time = time.time()
    
    last_predicted_label = current_label

def main():
    global last_predicted_label, current_action_start_time, action_history_log

    # --- AYARLANACAK VIDEO KAYNAĞI ---
    VIDEO_SOURCE = 'pose_classification/video.mp4'
    # VIDEO_SOURCE = 0 # Canlı kamera için
    # --- AYARLAMALAR BİTTİ ---

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Hata: Video kaynağı '{VIDEO_SOURCE}' açılamadı.")
        return

    # Hareket geçmişi ve stabilizasyon için deque
    action_history = deque(maxlen=ACTION_HISTORY_BUFFER_SIZE)
    last_stable_action = "Tanımlanıyor..."
    
    resized_window = False

    print("--- 3 Hareketli Aksiyon Tespiti Başlatılıyor ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not resized_window:
            h, w, _ = frame.shape
            scale_w = MAX_DISPLAY_WIDTH / w
            scale_h = MAX_DISPLAY_HEIGHT / h
            scale = min(scale_w, scale_h, 1.0)
            display_w = int(w * scale)
            display_h = int(h * scale)
            cv2.namedWindow('3 Action Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('3 Action Detection', display_w, display_h)
            resized_window = True

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        pose_results = pose_detector.process(image)
        hands_results = hands_detector.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        instant_action = analyze_pose(pose_results.pose_landmarks, hands_results.multi_hand_landmarks)
        action_history.append(instant_action)

        # Hareket yumuşatma ve kararlı hareket tespiti
        action_counts = {}
        for action in action_history:
            if action != "Unknown":
                action_counts.setdefault(action, 0)
                action_counts[action] += 1
        
        pose_label = last_stable_action
        
        if "Clapping" in action_counts and action_counts["Clapping"] >= MIN_CONFIDENT_FRAMES:
            pose_label = "Clapping"
        elif "Sitting" in action_counts and action_counts["Sitting"] >= MIN_CONFIDENT_FRAMES:
            pose_label = "Sitting"
        elif "Standing" in action_counts and action_counts["Standing"] >= MIN_CONFIDENT_FRAMES:
            pose_label = "Standing"
        elif action_counts:
            most_common_action = max(action_counts, key=action_counts.get)
            if action_counts[most_common_action] >= 2:
                pose_label = most_common_action
        
        if pose_label != "Unknown" and action_counts.get(pose_label, 0) >= MIN_CONFIDENT_FRAMES:
            log_action_if_changed(pose_label)
            last_stable_action = pose_label
        elif pose_label == "Unknown" and not action_counts:
            log_action_if_changed(pose_label)
            last_stable_action = "Unknown"

        # Çizimler
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                         mp_drawing_styles.get_default_hand_landmarks_style(),
                                         mp_drawing_styles.get_default_hand_connections_style())

        # Bilgileri ekrana yazdır
        duration_display = time.time() - current_action_start_time
        display_text = f'Aksiyon: {last_predicted_label} | Sure: {duration_display:.1f}s'
        cv2.putText(image, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('3 Action Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Çıkışta son hareketi logla
    log_action_if_changed("Program Sonlandı")
    
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    hands_detector.close()

    # Log dosyasını kaydet
    with open(LOG_FILE, 'w') as f:
        import json
        json.dump(action_history_log, f, indent=4)
    print(f"\nDavranış geçmişi '{LOG_FILE}' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()