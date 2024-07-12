import os, sys, time, json
import copy, cv2
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
                    keypoint = kp_coord + [2] if kp_score >= 0.5 else kp_coord + [1]
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

def convert_dataset_golfpose(image_prefix, input_file, output_file, pad_ratio=0.0):
    coco_data = json.loads(open(input_file, 'r').read())

    new_categories = list()
    for category in coco_data['categories']:
        if category['name'] == 'golf_pose':
            new_category = copy.deepcopy(category)
            new_category['name'] = 'person'
            new_category['id'] = 1
            new_categories.append(new_category)
            break
    coco_data['categories'] = new_categories

    image_infos = coco_data['images']
    image_infos = dict([(image_info['id'], image_info) for image_info in image_infos])

    num_invalids = 0
    new_annotations = list()
    for annotation in coco_data['annotations']:
        new_annotation = copy.deepcopy(annotation)
        image_info = image_infos[new_annotation['image_id']]
        img_w, img_h = image_info['width'], image_info['height']
        keypoints = new_annotation['keypoints']
        for i in range(30):
            if keypoints[i*3] == 0 and keypoints[i*3+1] == 0:
                keypoints[i*3+2] = 0
        bbox = new_annotation['bbox']
        def check_bbox(bbox, keypoints, image_info):
            if bbox[2] < 10 or bbox[3] < 10:
                print(f'Invalid annotation: {bbox}')
                return None
            return bbox
        bbox = check_bbox(bbox, keypoints, image_info)
        if bbox is None:
            num_invalids += 1
            continue
        if pad_ratio > 0.0:
            width, height = bbox[2:4]
            bbox[0] -= width * pad_ratio/2
            bbox[1] -= height * pad_ratio/2
            bbox[2] += width * pad_ratio
            bbox[3] += height * pad_ratio
        limit_bbox(bbox, img_w, img_h)
        new_annotation['category_id'] = 1
        new_annotation['bbox'] = bbox
        new_annotations.append(new_annotation)
    print(f'Number of invalid annotations: {num_invalids}')
    coco_data['annotations'] = new_annotations

    json_str = json.dumps(coco_data, ensure_ascii=False, indent=4)
    open(output_file, 'w').write(json_str)

def convert_dataset_golfclub(image_prefix, input_file, output_file, pad_ratio=0.0, 
                                  verbose=False):
    coco_data = json.loads(open(input_file, 'r').read())

    new_categories = list()
    for category in coco_data['categories']:
        if category['name'] == 'golf_pose':
            new_category = copy.deepcopy(category)
            new_category['name'] = 'golf_club'
            new_category['id'] = 1
            new_category['keypoints'] = ['golf_club_head', 'golf_club_tail']
            new_category['skeleton'] = [['golf_club_head', 'golf_club_tail']]
            new_categories.append(new_category)
            break
    coco_data['categories'] = new_categories

    image_infos = coco_data['images']
    image_infos = dict([(image_info['id'], image_info) for image_info in image_infos])

    new_annotations = list()
    num_invalids = 0
    for annotation in coco_data['annotations']:
        new_annotation = copy.deepcopy(annotation)
        image_info = image_infos[new_annotation['image_id']]
        img_w, img_h = image_info['width'], image_info['height']
        keypoints = new_annotation['keypoints']
        for i in range(30):
            if keypoints[i*3] == 0 and keypoints[i*3+1] == 0:
                keypoints[i*3+2] = 0
        keypoints = new_annotation['keypoints'][84:90]
        def check_bbox(keypoints, image_info):
            if keypoints[2] > 0 and keypoints[5] > 0:
                xs = min(keypoints[0], keypoints[3])
                xe = max(keypoints[0], keypoints[3])
                ys = min(keypoints[1], keypoints[4])
                ye = max(keypoints[1], keypoints[4])
                bw, bh = xe-xs, ye-ys
                bbox = [xs-2, ys-2, bw+4, bh+4]
                if bw < 10 and bh < 10:
                    print(f'Invalid annotation: {bbox}')
                    if verbose:
                        file_path = os.path.join('/data/ModelTrainData', image_info['file_name'])
                        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                        pt1 = [int(bbox[0]), int(bbox[1])]
                        pt2 = [int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])]
                        cv2.rectangle(image, pt1, pt2, (255,0,0), 2)
                        pt1 = [int(x) for x in keypoints[0:2]]
                        pt2 = [int(x) for x in keypoints[3:5]]
                        cv2.line(image, pt1, pt2, (0,0,255), 2)
                        cv2.imshow('test', image)
                        cv2.waitKey(0)
                    return None
                return bbox
            return None
        bbox = check_bbox(keypoints, image_info)
        if bbox is None:
            num_invalids += 1
            continue
        if pad_ratio > 0.0:
            width, height = bbox[2:4]
            bbox[0] -= width * pad_ratio/2
            bbox[1] -= height * pad_ratio/2
            bbox[2] += width * pad_ratio
            bbox[3] += height * pad_ratio
        limit_bbox(bbox, img_w, img_h)
        new_annotation['category_id'] = 1
        new_annotation['keypoints'] = keypoints
        new_annotation['num_keypoints'] = 2
        new_annotation['bbox'] = bbox
        new_annotation['area'] = bbox[2] * bbox[3]
        new_annotations.append(new_annotation)
    print(f'Number of invalid annotations: {num_invalids}')
    coco_data['annotations'] = new_annotations

    json_str = json.dumps(coco_data, ensure_ascii=False, indent=4)
    open(output_file, 'w').write(json_str)

def convert_dataset_halpe28(image_prefix, input_file, output_file, pad_ratio=0.0, verbose=False):
    coco_data = json.loads(open(input_file, 'r').read())

    new_categories = list()
    for category in coco_data['categories']:
        if category['name'] == 'golf_pose':
            new_category = copy.deepcopy(category)
            new_category['name'] = 'person'
            new_category['id'] = 1
            new_category['keypoints'] = [str(i) for i in range(28)]
            new_category['skeleton'] = [line for line in new_category['skeleton'] \
                                        if line != [30, 29]]
            new_categories.append(new_category)
            break
    coco_data['categories'] = new_categories

    image_infos = coco_data['images']
    image_infos = dict([(image_info['id'], image_info) for image_info in image_infos])

    new_annotations = list()
    num_invalids = 0
    for annotation in coco_data['annotations']:
        new_annotation = copy.deepcopy(annotation)
        image_info = image_infos[new_annotation['image_id']]
        img_w, img_h = image_info['width'], image_info['height']
        keypoints = new_annotation['keypoints']
        for i in range(30):
            if keypoints[i*3] == 0 and keypoints[i*3+1] == 0:
                keypoints[i*3+2] = 0
        keypoints = new_annotation['keypoints'][0:84]
        bbox = new_annotation['bbox']
        def check_bbox(bbox, keypoints, image_info):
            if bbox[2] < 10 or bbox[3] < 10:
                print(f'Invalid annotation: {bbox}')
                return None
            return bbox
        bbox = check_bbox(bbox, keypoints, image_info)
        if bbox is None:
            num_invalids += 1
            continue
        if pad_ratio > 0.0:
            width, height = bbox[2:4]
            bbox[0] -= width * pad_ratio/2
            bbox[1] -= height * pad_ratio/2
            bbox[2] += width * pad_ratio
            bbox[3] += height * pad_ratio
        limit_bbox(bbox, img_w, img_h)
        new_annotation['category_id'] = 1
        new_annotation['keypoints'] = keypoints
        new_annotation['num_keypoints'] = 28
        new_annotation['bbox'] = bbox
        new_annotations.append(new_annotation)
    print(f'Number of invalid annotations: {num_invalids}')
    coco_data['annotations'] = new_annotations

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
        convert_dataset_golfpose(args.data_prefix, args.input_path, args.output_path)
    elif args.work_mode == 3:
        convert_dataset_golfclub(args.data_prefix, args.input_path, args.output_path)
    elif args.work_mode == 4:
        convert_dataset_halpe28(args.data_prefix, args.input_path, args.output_path)
    elif args.work_mode == 5:
        split_coco_dataset(args.input_path, args.data_prefix, args.pad_ratio)

if __name__ == '__main__':
    main()
