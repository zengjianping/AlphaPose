import os, sys, time, json
import numpy as np
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--work-mode', type=int, default=0, help='Work mode')
    parser.add_argument('--input-path', type=str, default=None, help='input path')
    parser.add_argument('--output-path', type=str, default=None, help='output path')

    args = parser.parse_args()
    return args

def preprocess_coco_dataset(input_file):
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

def process_coco_dataset(input_file, output_file):
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

def main():
    args = parse_args()
    if args.work_mode == 0:
        preprocess_coco_dataset(args.input_path)
    elif args.work_mode == 1:
        process_coco_dataset(args.input_path, args.output_path)

if __name__ == '__main__':
    main()
