import os, sys, time, json
import copy
import numpy as np
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--work-mode', type=int, default=0, help='Work mode')
    parser.add_argument('--input-path', type=str, default=None, help='input path')
    parser.add_argument('--output-path', type=str, default=None, help='output path')
    parser.add_argument('--data-prefix', type=str, default=None, help='data prefix')
    parser.add_argument('--pad-ratio', type=float, default=0.0, help='pad ratio')

    args = parser.parse_args()
    return args

def generate_image_list(input_file):
    coco_data = json.loads(open(input_file, 'r').read())
    image_infos = coco_data['images']

    data_dir = os.path.dirname(input_file)
    list_file = os.path.join(data_dir, 'image_list.txt')
    fil = open(list_file, 'w')

    for image_idx, image_info in enumerate(image_infos):
        print(f'Processing {image_idx+1}/{len(image_infos)}...')
        image_name = image_info['file_name']
        fil.write(f'{image_name}\n')
    fil.close()

def convert_algodet_result(input_file, output_file):
    coco_data = json.loads(open(input_file, 'r').read())
    coco_data.pop('annotations')

    result_file = 'examples/res/alphapose-results.json'
    result_data = json.loads(open(result_file, 'r').read())

    det_results = dict()
    for detection in result_data:
        image_name = detection['image_name']
        if image_name not in det_results:
            det_results[image_name] = list()
        det_results[image_name].append(detection)

    annot_id = 0
    category_id = 0
    for category in coco_data['categories']:
        if category['name'] == 'golf_pose':
            category_id = category['id']

    annotations = list()
    image_infos = coco_data['images']
    kp_indice = list(range(26)) + [103, 124]

    for image_idx, image_info in enumerate(image_infos):
        print(f'Processing {image_idx+1}/{len(image_infos)}...')

        image_name = image_info['file_name']
        if image_name in det_results:
            for region in det_results[image_name]:
                if region['score'] < 0.5:
                    continue
                annot_id += 1
                bbox = region['box'].copy()
                kp_coords = list()
                kp_scores = list()
                for idx in kp_indice:
                    kp_coords.append(region['keypoints'][idx*3:idx*3+2])
                    kp_scores.append(region['keypoints'][idx*3])

                keypoints = list()
                for kp_coord, kp_score in zip(kp_coords, kp_scores):
                    keypoint = kp_coord + [2] if kp_score >= 0.5 else [0, 0, 0]
                    keypoints.extend(keypoint)
                keypoints.extend([0, 0, 0] * 2)

                annot_obj = {
                    "id": annot_id,
                    "image_id": image_info['id'],
                    "category_id": category_id,
                    "segmentation": [],
                    "area": bbox[2] * bbox[3],
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {
                        "occluded": False,
                        "keyframe": False
                    },
                    "keypoints": keypoints,
                    "num_keypoints": len(keypoints) // 3
                }
                annotations.append(annot_obj)
    coco_data['annotations'] = annotations

    json_str = json.dumps(coco_data, ensure_ascii=False, indent=4)
    open(output_file, 'w').write(json_str)

def limit_bbox(bbox, img_w, img_h):
    left, top = bbox[0:2]
    width, height = bbox[2:4]
    right = left + width
    bottom = top + height
    left = max(left, 0)
    right = min(right, img_w)
    top = max(top, 0)
    bottom = min(bottom, img_h)
    bbox[0] = left
    bbox[1] = top
    bbox[2] = right - left
    bbox[3] = bottom - top

def convert_dataset_pose2det(input_file, output_file, pad_ratio=0.0):
    coco_data = json.loads(open(input_file, 'r').read())

    categories = coco_data['categories']
    categories = [category for category in categories if category['id'] == 1]
    coco_data['categories'] = categories

    image_infos = coco_data['images']
    image_infos = dict([(image_info['id'], image_info) for image_info in image_infos])

    annotations = coco_data['annotations']
    for annotation in annotations:
        annotation.pop('keypoints')
        annotation['category_id'] = 1
        bbox = annotation['bbox']
        if pad_ratio > 0.0:
            width, height = bbox[2:4]
            bbox[0] -= width * pad_ratio/2
            bbox[1] -= height * pad_ratio/2
            bbox[2] += width * pad_ratio
            bbox[3] += height * pad_ratio
        image_info = image_infos[annotation['image_id']]
        img_w, img_h = image_info['width'], image_info['height']
        limit_bbox(bbox, img_w, img_h)

    json_str = json.dumps(coco_data, ensure_ascii=False, indent=4)
    open(output_file, 'w').write(json_str)

def split_coco_dataset(input_file, image_prefix, pad_ratio=0.0):
    coco_data = json.loads(open(input_file, 'r').read())
    image_infos = coco_data['images']
    annotations = coco_data['annotations']

    for image_info in image_infos:
        image_info['file_name'] = image_info['file_name'][len(image_prefix):]
    image_dict = dict([(image_info['id'], image_info) for image_info in image_infos])

    num_train_images = int(len(image_infos) * 0.8)
    train_image_infos = image_infos[0:num_train_images]
    val_image_infos = image_infos[num_train_images:]
    train_image_ids = set([info['id'] for info in train_image_infos])

    train_annotations = list()
    val_annotations = list()
    for annotation in annotations:
        bbox = annotation['bbox']
        if pad_ratio > 0.0:
            width, height = bbox[2:4]
            bbox[0] -= width * pad_ratio/2
            bbox[1] -= height * pad_ratio/2
            bbox[2] += width * pad_ratio
            bbox[3] += height * pad_ratio
        image_info = image_dict[annotation['image_id']]
        img_w, img_h = image_info['width'], image_info['height']
        limit_bbox(bbox, img_w, img_h)
        if annotation['image_id'] in train_image_ids:
            train_annotations.append(annotation)
        else:
            val_annotations.append(annotation)
    
    train_coco_data = copy.deepcopy(coco_data)
    train_coco_data['images'] = train_image_infos
    train_coco_data['annotations'] = train_annotations
    val_coco_data = copy.deepcopy(coco_data)
    val_coco_data['images'] = val_image_infos
    val_coco_data['annotations'] = val_annotations

    data_dir = os.path.dirname(input_file)
    train_file = os.path.join(data_dir, 'train.json')
    fil = open(train_file, 'w', encoding='utf-8')
    json.dump(train_coco_data, fil, ensure_ascii=False, indent=4)
    fil.close()

    val_file = os.path.join(data_dir, 'val.json')
    fil = open(val_file, 'w', encoding='utf-8')
    json.dump(val_coco_data, fil, ensure_ascii=False, indent=4)
    fil.close()

def main():
    args = parse_args()
    if args.work_mode == 0:
        generate_image_list(args.input_path)
    elif args.work_mode == 1:
        convert_algodet_result(args.input_path, args.output_path)
    elif args.work_mode == 2:
        convert_dataset_pose2det(args.input_path, args.output_path)
    elif args.work_mode == 3:
        split_coco_dataset(args.input_path, args.data_prefix, args.pad_ratio)

if __name__ == '__main__':
    main()
