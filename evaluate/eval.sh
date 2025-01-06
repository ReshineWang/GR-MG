#cd /PATH/TO/GR_MG
export EPOCHS=(47)
export CKPT_DIR="/home/wangrx/Projects/GR-MG/resources"
export SD_CKPT="/home/wangrx/Projects/GR-MG/resources/goal_gen.ckpt"
export MESA_GL_VERSION_OVERRIDE=3.3
echo $EPOCHS
echo $CKPT_DIR
sudo chmod 777 -R ${CKPT_DIR}

export COUNTER=-1
# Use a for loop to iterate through a list
for epoch in "${EPOCHS[@]}"; do
    export COUNTER=$((${COUNTER} + 1))
    export CUDA_VISIBLE_DEVICES=${COUNTER}
    python3 evaluate/eval_robotwin.py \
        --ckpt_dir ${CKPT_DIR} \
        --epoch ${epoch} \
        --ip2p_ckpt_path ${SD_CKPT} \
        --config_path ${@:1} &
done
wait