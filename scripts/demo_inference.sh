set -x

export PYTHONPATH=.:$PYTHONPATH

#CONFIG=configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml
#CKPT=pretrained_models/fast_res50_256x192.pth

CONFIG=configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-dcn-combined.yaml
CKPT=pretrained_models/multi_domain_fast50_dcn_combined_256x192.pth

#CONFIG=configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml
#CKPT=pretrained_models/multi_domain_fast50_regression_256x192.pth

INDIR=examples/demo
OUTDIR=examples/res

#VIDEO=datas/golf_videos/test01.mp4
INDIR=/data/ModelTrainData
#IMGLIST=datas/golf_annotations/ezgolf_task_20240418/annotations/image_list.txt
IMGLIST=datas/golf_annotations/golfdb/annotations/image_list.txt

python scripts/demo_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --detector yolo \
    --detbatch 1 \
    --posebatch 1 \
    --qsize 8 \
    --indir ${INDIR} \
    --list ${IMGLIST} \
    --outdir ${OUTDIR}

#    --save_img \
#    --video ${VIDEO} \
#    --save_video \
