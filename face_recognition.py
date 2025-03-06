import cv2
import numpy as np
import pickle
import threading
import queue
from insightface.app import FaceAnalysis

# Nạp dữ liệu khuôn mặt từ file
with open("face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

# Khởi tạo InsightFace
face_app = FaceAnalysis(providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Mở camera (0 = camera laptop)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Queue lưu trữ frames từ camera
frame_queue = queue.Queue(maxsize=1)  # Giữ tối đa 1 frame để giảm độ trễ

# 📌 **Luồng đọc camera liên tục**
def camera_reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.empty():
            frame_queue.get_nowait()  # Giữ frame mới nhất
        frame_queue.put(frame)

# Chạy thread để đọc camera song song
thread = threading.Thread(target=camera_reader, daemon=True)
thread.start()

while True:
    if frame_queue.empty():
        continue

    frame = frame_queue.get()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB

    # Nhận diện khuôn mặt
    faces = face_app.get(rgb_frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        face_embedding = face.embedding

        best_match = None
        best_score = -1
        matched_info = None

        # Duyệt qua từng người trong face_db
        for student_id, data in face_db.items():
            embeddings_list = data["embeddings"]  # Danh sách embeddings của 1 người
            max_similarity = max(
                np.dot(face_embedding, db_emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(db_emb))
                for db_emb in embeddings_list
            )

            if max_similarity > best_score:
                best_score = max_similarity
                best_match = student_id
                matched_info = data

        # Nếu tìm thấy khuôn mặt khớp
        if best_score > 0.5:
            text_id = f"MaSV: {best_match}"
            text_name = f"Ho Ten: {matched_info['name']}"
            text_class = f"Lop: {matched_info['class']}"
            text_sim = f"Similarity: {best_score:.2f}"  

            cv2.putText(frame, text_id, (bbox[0], bbox[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, text_name, (bbox[0], bbox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, text_class, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, text_sim, (bbox[0], bbox[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Khong xac dinh", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Vẽ khung nhận diện
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    # Hiển thị video
    cv2.imshow("Face Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
