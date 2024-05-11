set -x

CONFIG=configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml
CKPT=pretrained_models/pretrained_w_cam.pth
VIDEO=datas/golf_videos/test01.mp4
OUTDIR=examples/res_3d

python scripts/demo_3d_inference.py \
    --cfg ${CONFIG} \
    --checkpoint ${CKPT} \
    --video ${VIDEO} \
    --outdir ${OUTDIR} \
    --detector yolo  --save_img --save_video
