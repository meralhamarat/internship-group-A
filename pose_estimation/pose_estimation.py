import cv2
import mediapipe as mp
import os
import re # Doğal sıralama için re (regular expression) modülünü içe aktarıyoruz

# --- MediaPipe Kurulumu ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- Yapılandırma ---
IMAGE_FOLDER = "images"  # Resimlerinizin bulunduğu klasör
MAX_DISPLAY_DIM = 800    # Görüntünün ekranda gösterileceği maksimum boyut (genişlik veya yükseklik)

# --- Poz Algılama Modeli ---
pose = mp_pose.Pose(static_image_mode=True)

# --- Yardımcı Fonksiyonlar ---
def resize_for_display(image, max_dim):
    """Görüntüyü belirtilen maksimum boyuta göre yeniden boyutlandırır."""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image

def natural_sort_key(s):
    """Dosya adlarını doğal (sayısal) olarak sıralamak için anahtar oluşturur."""
    # Metni sayısal ve metinsel parçalara ayırırız
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def get_image_list(folder):
    """Belirtilen klasördeki resim dosyalarını doğal (sayısal) olarak listeler."""
    if not os.path.exists(folder):
        print(f"Hata: Resim klasörü '{folder}' bulunamadı. Lütfen '{folder}' adında bir klasör oluşturun.")
        return []
    images = [img for img in os.listdir(folder) if img.lower().endswith((".jpg", ".png", ".jpeg"))]
    images.sort(key=natural_sort_key) # Dosyaları doğal (sayısal) sıraya göre sıralarız
    return images

def display_keypoint_names(image, results, keypoints_to_show):
    """Belirli kilit noktaların isimlerini resim üzerine yazar."""
    if results.pose_landmarks:
        for name, landmark_enum in keypoints_to_show.items():
            landmark = results.pose_landmarks.landmark[landmark_enum.value]
            h, w, _ = image.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.putText(image, name, (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return image

# --- Ana Uygulama Mantığı ---
def run_simple_pose_viewer():
    image_files = get_image_list(IMAGE_FOLDER)

    if not image_files:
        print("Klasörde resim bulunamadı. Lütfen 'images' klasörüne resim ekleyin.")
        return

    current_image_index = 0

    keypoints_to_show = {
        "Nose": mp_pose.PoseLandmark.NOSE,
        "L_Wrist": mp_pose.PoseLandmark.LEFT_WRIST,
        "R_Wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
        "L_Shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
        "R_Shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
        "L_Elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
        "L_Ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
        "R_Ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
    }

    cv2.namedWindow("Pose Viewer", cv2.WINDOW_AUTOSIZE)

    while True:
        # Resim dizin sınır kontrolü
        current_image_index = max(0, min(current_image_index, len(image_files) - 1))

        img_path = os.path.join(IMAGE_FOLDER, image_files[current_image_index])
                
        image = cv2.imread(img_path)

        if image is None:
            print(f"UYARI: Resim yüklenemedi veya bulunamadı: {img_path}. Sonraki resme geçiliyor.")
            current_image_index += 1 # Yüklenemezse otomatik olarak bir sonraki resme atla
            continue # Döngünün başına dön ve yeni dizinle tekrar dene

        display_image = resize_for_display(image.copy(), MAX_DISPLAY_DIM)
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            display_image = display_keypoint_names(display_image, results, keypoints_to_show)

        # Bilgi metni sadece indeks ve toplam sayıyı gösterecek şekilde düzenlendi
        info_text = f"{current_image_index + 1}/{len(image_files)}"
        cv2.putText(display_image, info_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(display_image, "Tuslar: A/D (Onceki/Sonraki), ESC (Cikis)", (20, display_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow("Pose Viewer", display_image)

        key = cv2.waitKey(0) # Kullanıcı tuşa basana kadar bekler

        if key == 27:  # ESC tuşu
            break
        elif key == ord('d'):  # 'd' tuşu ile sonraki resim
            current_image_index += 1
        elif key == ord('a'):  # 'a' tuşu ile önceki resim
            current_image_index -= 1

    pose.close()
    cv2.destroyAllWindows()

# Uygulamayı başlat
if __name__ == "__main__":
    run_simple_pose_viewer()