import cv2
import dlib
import pyttsx3
from scipy.spatial import distance

# KHỞI TẠO pyttsx3 ĐỂ CÓ THỂ PHÁT ÂM THANH CẢNH BÁO
engine = pyttsx3.init()

# CẤU HÌNH CAMERA ĐỂ 1, BẠN CÓ THỂ CHỌN 0 THAY VÌ 1
cap = cv2.VideoCapture(1)

# NHẬN DẠNG KHUÔN MẶT VÀ TÌM GÓC NHÌN ĐỂ NHẬN DIỆN MẮT
face_detector = dlib.get_frontal_face_detector()

# ĐƯỜNG DẪN ĐẾN FILE .DAT (FILE ĐỂ NHẬN DIỆN ĐIỂM TRÊN KHUÔN MẶT)
dlib_facelandmark = dlib.shape_predictor(
    "C:\\\\Users\\\\KHANH TAM\\\\Documents\\\\Python\\\\Nhan_dien_khuon_mat\\\\shape_predictor_68_face_landmarks.dat")

# HÀM TÍNH TOÁN TỶ LỆ KHỐI LƯỢNG CHO MẮT BẰNG CÁCH SỬ DỤNG HÀM KHOẢNG CÁCH EUCLIDEAN
def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
    return aspect_ratio_Eye

# VÒNG LẶP CHÍNH CHẠY ĐIỀU NÀY SẼ CHẠY TẤT CẢ CÁC THỜI GIAN CHO ĐẾN KHI CHƯƠNG TRÌNH BỊ KẾT THÚC BỞI NGƯỜI DÙNG
while True:
    null, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray_scale)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        leftEye = []
        rightEye = []

        # ĐIỂM PHÂN PHỐI CHO MẮT TRÁI TRONG FILE .DAT LÀ TỪ 42 ĐẾN 47
        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        # ĐIỂM PHÂN PHỐI CHO MẮT PHẢI TRONG FILE .DAT LÀ TỪ 36 ĐẾN 41
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        # TÍNH TOÁN TỶ LỆ KHỐI LƯỢNG CHO MẮT TRÁI VÀ PHẢI
        right_Eye = Detect_Eye(rightEye)
        left_Eye = Detect_Eye(leftEye)
        Eye_Rat = (left_Eye+right_Eye)/2

        # LÀM TRÒN GIÁ TRỊ
        Eye_Rat = round(Eye_Rat, 2)

        # HIỂN THỊ KẾT QUẢ TRÊN KHUNG HÌNH
        cv2.putText(frame, "Eye Ratio: {}".format(Eye_Rat), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # NẾU TỶ LỆ KHỐI LƯỢNG MẮT NHỎ HƠN 0,15 THÌ BẠN ĐANG SẮP BUỒN NGỦ
        if Eye_Rat < 0.15:
            cv2.putText(frame, "Dang buon ngu kia", (15, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Thuc day mau", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # LƯU TRỮ ÂM THANH CẢNH BÁO VÀ PHÁT NÓ
            engine.say('wake up wake up wake up wake up')
            engine.runAndWait()

    # HIỂN THỊ KHUNG HÌNH
    cv2.imshow("Phan mem nhan dien buon ngu", frame)

    # ĐỢI 1 MILISECOND VÀ XÓA BỘ NHỚ ĐỆM
    key = cv2.waitKey(1) & 0xFF

    # NẾU NGƯỜI DÙNG ẤN Q THÌ CHƯƠNG TRÌNH SẼ DỪNG
    if key == ord("q"):
        break

# ĐÓNG TẤT CẢ CÁC CỬA SỔ
cv2.destroyAllWindows()
cap.stop()
