# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import re
import copy
from copy import deepcopy
import os
import yaml
# This is for using the locally installed repo clone when using slurm
import matplotlib.pyplot as plt
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition
)
import hydra
import numpy as np
import importlib
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
from utils.utils import print_and_save
from wrapper.model_wrapper import CustomModel
from goal_gen.evaluate import IP2PEvaluation
import traceback
    
logger = logging.getLogger(__name__)
EP_LEN = 360
NUM_SEQUENCES = 1000
SAVE_DIR = None
FAIL_COUNTER=0

def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def make_env(dataset_path, observation_space, device_id):
    val_folder = Path(dataset_path) / "validation"
    # insert your own env wrapper
    from wrapper.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    device = torch.device('cuda', device_id)
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env

def test_policy(model, Demo_class, env_args, eval_sr_path, eval_result_path, ip2p_model):
    """Run this function to evaluate a model on the Robotwin challenge."""
    expert_check = True
    Demo_class.suc = 0
    Demo_class.test_num =0
    model.reset()
    st_seed=100000
    test_num=20

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    now_seed = st_seed
    while succ_seed < test_num:
        render_freq = env_args['render_freq']
        env_args['render_freq'] = 0
        
        if expert_check:
            try:
                Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** env_args)
                Demo_class.play_once()
                Demo_class.close()
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(' -------------')
                print('Error: ', stack_trace)
                print(' -------------')
                Demo_class.close()
                now_seed += 1
                env_args['render_freq'] = render_freq
                print('error occurs !')
                continue

        if (not expert_check) or ( Demo_class.plan_success and Demo_class.check_success() ):
            succ_seed +=1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            env_args['render_freq'] = render_freq
            continue

        env_args['render_freq'] = render_freq

        Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** env_args)
        Demo_class.apply_GR_MG(model, ip2p_model, env_args)

        now_id += 1
        Demo_class.close()
        if Demo_class.render_freq:
            Demo_class.viewer.close()

        print(f"block_hammer_beat success rate: {Demo_class.suc}/{Demo_class.test_num}, current seed: {now_seed}\n")
        Demo_class._take_picture()
        now_seed += 1

    return now_seed, Demo_class.suc
    
def evaluate_policy(model, env, eval_sr_path, eval_result_path, ip2p_model):
    """Run this function to evaluate a model on the CALVIN challenge."""
    conf_dir = Path("./calvin/calvin_models/conf")
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_sequences = get_sequences(NUM_SEQUENCES)
    results = []
    sequence_i = 0
    for index,(initial_state, eval_sequence) in enumerate(eval_sequences):
        result= evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, sequence_i,ip2p_model)
        results.append(result)
        success_list = count_success(results)
        with open(eval_sr_path, 'a') as f:
            line =f"{sequence_i}/{NUM_SEQUENCES}: "
            for sr in success_list:
                line += f"{sr:.3f} | "
            sequence_i += 1
            line += "\n"
            f.write(line)

        if index%100==0 and index!=0: #save every 100 sequences
            print_and_save(results, eval_sequences[:index+1], eval_result_path[:-5]+f"_{index+1}"+".json", None)
    print_and_save(results, eval_sequences, eval_result_path, None)
    return results


def evaluate_sequence(Demo_class, model, task_checker, initial_state, eval_sequence, val_annotations, sequence_i,ip2p_model):
    """Evaluates a sequence of language instructions."""
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    Demo_class.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, subtask_i, sequence_i,ip2p_model)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout(env, model, task_oracle, subtask, val_annotations, subtask_i, sequence_i,ip2p_model):
    """Run the actual rollout on one subtask."""
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()
    debug_image=[]
    progress=0
    for i in range(EP_LEN):
        if i % 20 == 0:  # hardcode
            static_rgb = obs['rgb_obs']['rgb_static'] # (200, 200, 3)
            hand_rgb = obs['rgb_obs']['rgb_gripper']
            image_patch=[static_rgb]
            text_patch=[lang_annotation + f".And {progress}% of the instruction has been finished."]
            print(text_patch)
            goal_image=ip2p_model.inference(image_patch,text_patch)
            temp_image=[static_rgb,goal_image[0],hand_rgb]
            debug_image.append(temp_image)

        action,progress = model.step(obs,deepcopy(goal_image),[lang_annotation]) 
        obs, _, _, current_info = env.step(action)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            print("success!")
            return True
    print("fail!")
    
    global FAIL_COUNTER
    FAIL_COUNTER+=1
    if FAIL_COUNTER % 30 ==0:  # save every 30 failure cases
        length=len(debug_image) 
        fig, ax = plt.subplots(length, 2,figsize=(5.5, 46.58))
        for ax_ in ax.flat:
            # ax_.plot([1, 2, 3], [4, 5, 6])
            ax_.axis('off')  # 隐藏每个子图的刻度和边框
        for i in range(length):
            ax[i][0].imshow(debug_image[i][0])
            ax[i][1].imshow(debug_image[i][1])
            # ax[i][2].imshow(debug_image[i][2])
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(os.path.join(SAVE_DIR, f"{sequence_i}-{subtask_i}-{subtask}.png"),dpi=100)
        plt.close()
    return False

def get_camera_config(camera_type):
    camera_config_path = os.path.join('envs/_camera_config.yml')

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    print(args)
    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]

def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", default='/home/wangrx/Projects/GR-MG/resources/calvin_debug_dataset',
                        type=str, help="Path to the dataset root directory.")  # modify it before opensource
    # evaluation
    parser.add_argument('--config_path', type=str, default="", help='path to the policy config file')
    parser.add_argument('--ckpt_dir', type=str, default="",help="path to the policy ckpt file")
    parser.add_argument('--epoch', type=int,default=41, help="epoch index for evaluating")
    parser.add_argument('--device_id', default=0, type=int, help="CUDA device")
    parser.add_argument('--ip2p_ckpt_path', default="", type=str, help="ip2p ckpt path")
    usr_args = parser.parse_args()
    config_path = usr_args.config_path
    ckpt_dir = usr_args.ckpt_dir
    epoch = usr_args.epoch
    device_id = usr_args.device_id
    ip2p_ckpt_path=usr_args.ip2p_ckpt_path
    assert config_path != None
    # Load config file
    with open(config_path, 'r') as f:
        configs = json.load(f)
    with open(f'envs/block_hammer_beat.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader) 
    args['head_camera_type'] = "L515"
    head_camera_config = get_camera_config(args['head_camera_type'])
    args['head_camera_fovy'] = head_camera_config['fovy']
    args['head_camera_w'] = head_camera_config['w']
    args['head_camera_h'] = head_camera_config['h']
    head_camera_config = 'fovy' + str(args['head_camera_fovy']) + '_w' + str(args['head_camera_w']) + '_h' + str(args['head_camera_h'])
    
    wrist_camera_config = get_camera_config(args['wrist_camera_type'])
    args['wrist_camera_fovy'] = wrist_camera_config['fovy']
    args['wrist_camera_w'] = wrist_camera_config['w']
    args['wrist_camera_h'] = wrist_camera_config['h']
    wrist_camera_config = 'fovy' + str(args['wrist_camera_fovy']) + '_w' + str(args['wrist_camera_w']) + '_h' + str(args['wrist_camera_h'])

    front_camera_config = get_camera_config(args['front_camera_type'])
    args['front_camera_fovy'] = front_camera_config['fovy']
    args['front_camera_w'] = front_camera_config['w']
    args['front_camera_h'] = front_camera_config['h']
    front_camera_config = 'fovy' + str(args['front_camera_fovy']) + '_w' + str(args['front_camera_w']) + '_h' + str(args['front_camera_h'])

    # output camera config
    print('============= Camera Config =============\n')
    print('Head Camera Config:\n    type: '+ str(args['head_camera_type']) + '\n    fovy: ' + str(args['head_camera_fovy']) + '\n    camera_w: ' + str(args['head_camera_w']) + '\n    camera_h: ' + str(args['head_camera_h']))
    print('Wrist Camera Config:\n    type: '+ str(args['wrist_camera_type']) + '\n    fovy: ' + str(args['wrist_camera_fovy']) + '\n    camera_w: ' + str(args['wrist_camera_w']) + '\n    camera_h: ' + str(args['wrist_camera_h']))
    print('Front Camera Config:\n    type: '+ str(args['front_camera_type']) + '\n    fovy: ' + str(args['front_camera_fovy']) + '\n    camera_w: ' + str(args['front_camera_w']) + '\n    camera_h: ' + str(args['front_camera_h']))
    print('\n=======================================')


    # Get checkpoint path
    ckpt_path = None
    ckpt_files = os.listdir(ckpt_dir)
    for ckpt_file in ckpt_files:
        match = re.search(r'epoch=(\d+)', ckpt_file)
        if match:
            temp_epoch = int(match.group(1))
            if temp_epoch == epoch:
                ckpt_path = os.path.join(ckpt_dir, ckpt_file)
                break

    device = torch.device('cuda', device_id)
    model = CustomModel(
        ckpt_path=ckpt_path,
        configs=configs,
        device=device)
    #model = None
    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'], 
        'depth_obs': [], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']} 
    #env = make_env(args.dataset_path, observation_space, device_id) 
    task = class_decorator('block_hammer_beat')

    # Success rate and result files
    flag="opensourcesd"
    sub_dir=f"{flag}_{epoch}_epoch"
    # set a global variable
    global SAVE_DIR
    SAVE_DIR=os.path.join(ckpt_dir,sub_dir)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    sr_path = os.path.join(SAVE_DIR, f"success_rate.txt")
    result_path = os.path.join(SAVE_DIR, f"results.json")
    ip2p_model=IP2PEvaluation(ip2p_ckpt_path)
    #ip2p_model=None
    test_policy(
        model, 
        task,
        args,
        eval_sr_path=sr_path,
        eval_result_path=result_path,
        ip2p_model=ip2p_model)
    
if __name__ == "__main__":
    #from test_render import Sapien_TEST
    #Sapien_TEST()
    main()