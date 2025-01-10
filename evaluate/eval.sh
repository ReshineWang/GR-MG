#cd /PATH/TO/GR_MG
export EPOCHS=49
export CKPT_DIR="SAVE_PATH/policy/checkpoints/2025-01-08/train_policy"
export SD_CKPT="SAVE_PATH/goal_gen/checkpoints/2025-01-07/train_goal_gen/epoch=39-step=13160.ckpt"
export MESA_GL_VERSION_OVERRIDE=3.3
echo $EPOCHS
echo $CKPT_DIR
sudo chmod 777 -R ${CKPT_DIR}

export CUDA_VISIBLE_DEVICES=1  # 设置使用的 GPU 设备

python3 evaluate/eval_robotwin.py \
    --ckpt_dir ${CKPT_DIR} \
    --epoch ${EPOCHS} \
    --ip2p_ckpt_path ${SD_CKPT} \
    --config_path ${@:1}