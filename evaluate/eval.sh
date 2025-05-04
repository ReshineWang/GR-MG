#cd /PATH/TO/GR_MG
export EPOCHS=99
export CKPT_DIR="SAVE_PATH/policy/checkpoints/2025-03-15/dual_bottles_pick_hard"
export SD_CKPT="SAVE_PATH/goal_gen/checkpoints/2025-02-23/dual_bottles_pick_hard/epoch=49-step=13450.ckpt"
export MESA_GL_VERSION_OVERRIDE=3.3
echo $EPOCHS
echo $CKPT_DIR
sudo chmod 777 -R ${CKPT_DIR}

export CUDA_VISIBLE_DEVICES=2  # 设置使用的 GPU 设备

python3 evaluate/eval_robotwin.py \
    --ckpt_dir ${CKPT_DIR} \
    --epoch ${EPOCHS} \
    --ip2p_ckpt_path ${SD_CKPT} \
    --config_path ${@:1}