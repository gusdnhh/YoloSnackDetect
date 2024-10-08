import cv2
import matplotlib.pyplot as plt

def load_yolo_label(label_path):
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            elements = line.strip().split()
            class_labels.append(int(elements[0]))
            bboxes.append([float(x) for x in elements[1:]])
    return bboxes, class_labels

def draw_bboxes(image_path, label_path, output_image_path=None):
    # 이미지와 레이블 로드
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    bboxes, class_labels = load_yolo_label(label_path)
    
    # 바운딩 박스 그리기
    for bbox in bboxes:
        x_center, y_center, box_width, box_height = bbox
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Matplotlib으로 이미지 출력
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 이미지를 RGB로 변환
    plt.imshow(image_rgb)
    plt.axis('off')  # 축 제거
    plt.show()
    
    # 결과 이미지 저장 (옵션)
    if output_image_path:
        cv2.imwrite(output_image_path, image)

# 사용 예시
image_path = r'yolo_snack\trains\images\img_0571_aug_1.jpg'        # 증강된 이미지 파일 경로
label_path = r'yolo_snack\trains\labels\img_0571_aug_1.txt'        # 증강된 이미지의 YOLO 라벨 파일 경로
# output_image_path = 'data/augmented/labeled_image.jpg' # 결과 이미지를 저장할 경로 (옵션)

draw_bboxes(image_path, label_path)
