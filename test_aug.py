import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import glob

def load_yolo_label(label_path):
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            elements = line.strip().split()
            class_labels.append(int(elements[0]))
            bboxes.append([float(x) for x in elements[1:]])
    return bboxes, class_labels

def draw_bboxes(image, bboxes, class_labels):
    height, width, _ = image.shape
    for bbox, cls in zip(bboxes, class_labels):
        x_center, y_center, box_width, box_height = bbox
        x1 = int((x_center - box_width / 2) * width)
        y1 = int((y_center - box_height / 2) * height)
        x2 = int((x_center + box_width / 2) * width)
        y2 = int((y_center + box_height / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 클래스 레이블 추가 (옵션)
        cv2.putText(image, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return image

def show_images_with_bboxes(image_dir, label_dir):
    # 이미지 파일 목록 가져오기
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
    index = 0
    total_images = len(image_files)
    
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('Image Viewer')

    # 전체 화면으로 설정
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()

    def update_image():
        nonlocal index
        if index < 0:
            index = total_images - 1
        elif index >= total_images:
            index = 0
        
        image_path = image_files[index]
        base_name = os.path.basename(image_path)
        label_name = os.path.splitext(base_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)
        
        # 이미지와 라벨 로드
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러올 수 없습니다: {image_path}")
            return
        
        if os.path.exists(label_path):
            bboxes, class_labels = load_yolo_label(label_path)
            image = draw_bboxes(image, bboxes, class_labels)
        else:
            print(f"라벨 파일이 존재하지 않습니다: {label_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 현재 축을 지우고 새로운 이미지를 표시
        ax.clear()
        ax.imshow(image_rgb, aspect='auto')  # aspect='auto'로 설정하여 이미지 비율을 왜곡 가능하게 함
        ax.axis('off')
        ax.set_position([0, 0, 1, 1])  # 축을 전체 창에 맞게 조정

        fig.canvas.draw()

    def on_key(event):
        nonlocal index
        if event.key == 'right':
            index += 1
            update_image()
        elif event.key == 'left':
            index -= 1
            update_image()
        elif event.key == 'escape':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_image()
    plt.show()

# 사용 예시
image_dir = r'yolo_snack_dataset\trains\images'   # 이미지 폴더 경로
label_dir = r'yolo_snack_dataset\trains\labels'   # 라벨 폴더 경로

show_images_with_bboxes(image_dir, label_dir)
