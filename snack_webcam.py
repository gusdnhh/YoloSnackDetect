import cv2
from ultralytics import YOLO
import os
import time

def detect_snack_with_webcam():
    # 학습한 모델 불러오기
    model = YOLO('/Users/gusdnhh/Documents/yolo/snack_project/weights/best200.pt')  # best.pt 경로로 변경

    # 웹캠 열기
    cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 사용합니다
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    # 저장할 디렉토리 설정
    save_dir = 'testing_imgs'
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    while True:
        # 웹캠에서 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 모델을 사용해 간식 탐지 수행
        results = model(frame)

        # 감지된 객체에 바운딩 박스 그리기
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 바운딩 박스 좌표
                label = result.names[int(box.cls[0])]   # 클래스 이름
                conf = box.conf[0]                      # 신뢰도

                # 바운딩 박스와 라벨 추가
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 5)  # 글자 크기와 두께 조정

        # 프레임 출력
        cv2.imshow('YOLO Snack Detection', frame)

        # 키 입력 대기
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # 'q' 키를 누르면 종료
            break
        elif key == ord(' '):
            # 스페이스바를 누르면 프레임 저장
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"frame_{timestamp}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"프레임이 저장되었습니다: {filepath}")

    # 웹캠 및 창 닫기
    cap.release()
    cv2.destroyAllWindows()

# 함수 실행
detect_snack_with_webcam()
