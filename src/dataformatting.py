import os
import json

def convert_data_to_yolo_format(json_path, labels_dir):
    os.makedirs(labels_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_map = {img["id"]: img for img in data["images"]}
    ann_per_images = {img["id"]: [] for img in data["images"]}

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        ann_per_images[image_id].append(ann)
    
    for img_id, anns in ann_per_images.items():
        img_detail = image_map[img_id]
        img_w = img_detail['width']
        img_h = img_detail['height']
        file_name = img_detail["file_name"]
        txt_file = file_name.replace('.png', '.txt')

        label_path = os.path.join(labels_dir, txt_file)

        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in anns:
                x, y, w, h = ann["bbox"]
                class_id = ann["category_id"] - 1

                reformatted_x = (x + w / 2) / img_w
                reformatted_y = (y + h / 2) / img_h
                reformatted_w = w / img_w
                reformatted_h = h / img_h               
                f.write(f"{class_id} {reformatted_x} {reformatted_y} {reformatted_w} {reformatted_h}\n")

ROOT = "./98753/plastic_coco"

convert_data_to_yolo_format(json_path=f"{ROOT}/annotation/train.json", labels_dir=f"{ROOT}/labels/train")
convert_data_to_yolo_format(json_path=f"{ROOT}/annotation/test.json", labels_dir=f"{ROOT}/labels/test")
convert_data_to_yolo_format(json_path=f"{ROOT}/annotation/val.json", labels_dir=f"{ROOT}/labels/val")
