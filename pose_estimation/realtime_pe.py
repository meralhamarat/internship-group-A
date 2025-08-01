import cv2
import mediapipe as mp

# MediaPipe kurulum
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Kamera açılıyor (0 = varsayılan kamera)
cap = cv2.VideoCapture(0)

# Pose modelini başlat
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı.")
            break

        # BGR -> RGB dönüşümü
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False  # performans için (görüntü düzenlenmeyecek)

        # Poz tahmini işlemi
        results = pose.process(image_rgb)

        # Görüntüyü tekrar düzenlenebilir yap
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Eğer poz noktaları varsa çiz
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
            )

        cv2.imshow("Real-time Pose Estimation", image)

        # ESC tuşu ile çıkış
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
