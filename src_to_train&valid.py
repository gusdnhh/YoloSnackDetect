import os
import sys
import shutil
import random
import cv2
from collections import defaultdict
import albumentations as A

def compare_file_counts(directory):
    jpg_files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    jpg_count, txt_count = len(jpg_files), len(txt_files)
    print(f"JPG 파일 개수: {jpg_count}, TXT 파일 개수: {txt_count}")
    if jpg_count != txt_count:
        print("JPG와 TXT 파일의 개수가 다릅니다.")
        sys.exit()
    return True

def get_class_from_label(txt_file):
    with open(txt_file, 'r') as file:
        first_line = file.readline().strip()
        if first_line:
            return first_line.split()[0]
    return None

def load_yolo_label(label_path):
    bboxes = []
    with open(label_path, 'r') as file:
        for line in file.readlines():
            elements = line.strip().split()
            class_id = int(elements[0])
            bbox = list(map(float, elements[1:]))
            bboxes.append([class_id, *bbox])
    return bboxes

def save_yolo_label(label_path, bboxes):
    with open(label_path, 'w') as file:
        for bbox in bboxes:
            class_id, x, y, w, h = bbox
            file.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def split_and_augment_data(src_dir, train_dir, valid_dir, split_ratio=0.8, img_size=(256, 416), num_augmentations=6):
    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, 'labels'), exist_ok=True)
    
    # Train과 Valid에 사용할 변환 설정
    train_transform = A.Compose([
        A.Resize(*img_size),
        A.Rotate(limit=(-45, 45), p=0.5),
        A.RandomScale(scale_limit=0.5, p=0.5) ,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.RandomBrightnessContrast(p=0.3),
        A.Blur(blur_limit=3, p=0.1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    valid_transform = A.Compose([A.Resize(*img_size)], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    class_files = defaultdict(list)
    
    for file_name in os.listdir(src_dir):
        if file_name.endswith('.jpg'):
            label_file = os.path.join(src_dir, file_name.replace('.jpg', '.txt'))
            class_id = get_class_from_label(label_file)
            if class_id is not None:
                class_files[class_id].append(file_name)
    
    for class_id, files in class_files.items():
        random.shuffle(files)
        split_idx = int(len(files) * split_ratio)
        train_files, valid_files = files[:split_idx], files[split_idx:]
        
        # Train 데이터에 리사이즈 및 다양한 증강 적용
        for file_name in train_files:
            src_image = os.path.join(src_dir, file_name)
            src_label = os.path.join(src_dir, file_name.replace('.jpg', '.txt'))
            
            # 원본 이미지를 여러 번 증강하여 저장
            for i in range(num_augmentations):
                dst_image = os.path.join(train_dir, 'images', f"{os.path.splitext(file_name)[0]}_aug_{i}.jpg")
                dst_label = os.path.join(train_dir, 'labels', f"{os.path.splitext(file_name)[0]}_aug_{i}.txt")
                
                # 이미지 로드 및 증강 적용
                image = cv2.imread(src_image)
                bboxes = load_yolo_label(src_label)
                class_labels = [bbox[0] for bbox in bboxes]
                yolo_bboxes = [bbox[1:] for bbox in bboxes]
                
                transformed = train_transform(image=image, bboxes=yolo_bboxes, class_labels=class_labels)
                transformed_image, transformed_bboxes = transformed['image'], transformed['bboxes']
                transformed_class_labels = transformed['class_labels']
                
                # 증강된 이미지와 레이블 저장
                cv2.imwrite(dst_image, transformed_image)
                save_yolo_label(dst_label, [[cls, *bbox] for cls, bbox in zip(transformed_class_labels, transformed_bboxes)])
        
        # Valid 데이터에 리사이즈만 적용
        for file_name in valid_files:
            src_image = os.path.join(src_dir, file_name)
            src_label = os.path.join(src_dir, file_name.replace('.jpg', '.txt'))
            dst_image = os.path.join(valid_dir, 'images', file_name)
            dst_label = os.path.join(valid_dir, 'labels', file_name.replace('.jpg', '.txt'))
            
            # 이미지 로드 및 리사이즈만 적용
            image = cv2.imread(src_image)
            bboxes = load_yolo_label(src_label)
            class_labels = [bbox[0] for bbox in bboxes]
            yolo_bboxes = [bbox[1:] for bbox in bboxes]
            
            transformed = valid_transform(image=image, bboxes=yolo_bboxes, class_labels=class_labels)
            transformed_image, transformed_bboxes = transformed['image'], transformed['bboxes']
            transformed_class_labels = transformed['class_labels']
            
            # 변환된 이미지와 레이블 저장
            cv2.imwrite(dst_image, transformed_image)
            save_yolo_label(dst_label, [[cls, *bbox] for cls, bbox in zip(transformed_class_labels, transformed_bboxes)])

# 사용 예시
src_dir = 'yolo_snack_dataset/src_data'
train_dir = 'yolo_snack_dataset/trains'
valid_dir = 'yolo_snack_dataset/valids'
compare_file_counts(src_dir)
split_and_augment_data(src_dir, train_dir, valid_dir, split_ratio=0.8)
