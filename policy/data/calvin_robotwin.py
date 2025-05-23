# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from PIL import Image
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import time
import numpy as np
from data.utils import *
ACTION_POS_SCALE = 50
ACTION_ROT_SCALE = 20

# source: https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
class RandomShiftsAug(nn.Module):
    # perform random shifts independently for each image in the batch
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

class RandomShiftsSingleAug(nn.Module):
    #  perform same random shifts for each image in the batch
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate") # pad before shift
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift.repeat(n, 1, 1, 1)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class CalvinRobotwin_Policy(Dataset):
    def __init__(self,
                 data_dir,
                 config,
                 forward_n_min_max=[60, 62], 
                 mode='train',
                 subfolder='',
                 use_data_augmentation=True,
                 task_num=100000,
                 use_play=True,
                 use_labeled=True):
        super().__init__()
        '''
        seq_len: length of observation
        act_len: length of output action trajectroy
        forward_n_max:  the maximum interval of goal image
        use_data_augmentation: whether to use RandomShiftsSingleAug
        task_num: It determins the maximum number of trajectories that could be sampled from a trajectory. If you want to train with all the trajectories,
        just set a very great value. When you want to do fewshot experiments, you should set it as a specific value, such as 52( 10% of total data ).
        use_full: whether to use play data
        use_labeled: whether to use data annotated with language
        '''
        self.dataset_dir = os.path.join(data_dir, subfolder)
        self.seq_len = config["policy"]["seq_len"] 
        self.act_len = config["policy"]["act_len"]
        self.forward_n_min, self.forward_n_max = forward_n_min_max
        self.mode = mode
        self.use_data_augmentation = use_data_augmentation
        self.task_num = task_num
        self.use_play=use_play
        self.use_labeled=use_labeled

        self.action_dim = config["input"]["act_dim"]
        self.state_dim = config["input"]["state_dim"]  # ( x,y,z,r,p,y,gripper) 

        if mode == 'validate':
            tag = "validation"
        elif mode == 'train':
            tag = "training"
        else:
            raise ValueError("Mode must be either train or validate")
        
        # As we use MAE as Encoder, we need to resize and normalize
        self.input_size = (224, 224)
        self.clip_mean = (0.485, 0.456, 0.406) 
        self.clip_std = (0.229, 0.224, 0.225)

        if self.use_data_augmentation: # 随机平移数据增强
            self.static_rgb_preprocess_train = T.Compose([
                RandomShiftsSingleAug(pad=10), # 10 for static rgb
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
            self.hand_rgb_preprocess_train = T.Compose([
                RandomShiftsSingleAug(pad=4), # 4 for hand rgb
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
        else:
            self.static_rgb_preprocess_train = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
            self.hand_rgb_preprocess_train = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC),
                T.Normalize(self.clip_mean, self.clip_std)])
        
        self.static_rgb_preprocess_val = T.Compose([
            T.Resize(self.input_size, interpolation=Image.BICUBIC),
            T.Normalize(self.clip_mean, self.clip_std)])
        self.hand_rgb_preprocess_val = T.Compose([
            T.Resize(self.input_size, interpolation=Image.BICUBIC),
            T.Normalize(self.clip_mean, self.clip_std)])

        self.dataset_dir = os.path.join(self.dataset_dir)
        self._initialize()
        print(f'{len(self)} samples in total')

    def load(self,path):
        # Sometimes we can not immediately load the data when training in our GPU cluster. We set this protection mechanism
        t1=time.time()
        while True:
            try:
                result=np.load(path)
                break
            except:
                t2=time.time()
                if (t2-t1) >60:
                    assert False , "bad load"
                else:
                    continue
        return result


    def _initialize(self):
        """Generate the sequence index pair."""
        self.anns = np.load(
            os.path.join(self.dataset_dir, "lang_annotations", "auto_lang_ann.npy"), 
            allow_pickle=True).item()
        
        # (traj_index, start, end, act_end, traj_st)
        self.seq_tuple = []
        task_dict = {}
        n_trajs = len(self.anns['info']['indx'])
        print(f"{n_trajs} labeled trajectories!")

        if self.use_labeled: 
            for traj_idx in range(n_trajs):
                traj_st, traj_ed = self.anns['info']['indx'][traj_idx] # 获取一条轨迹的起始和结束帧
                traj_task = self.anns['language']['task'][traj_idx] # 获取一条轨迹的任务
                text=self.anns["language"]["ann"][traj_idx] # 获取一条轨迹的文本描述
                if (traj_ed - self.seq_len - self.act_len + 2) <= traj_st:
                    continue # exclude extremely short trajectory

                # sample trajectories based on task_num
                if traj_task not in task_dict:
                    task_dict[traj_task] = 1
                else:
                    task_dict[traj_task] = task_dict[traj_task] + 1
                if task_dict[traj_task] > self.task_num: # when arriving the maximum number, stop sampling from the task
                    continue # 确保每个任务的轨迹数量不超过 task_num
                # the action in the last frame is discarded
                for st in range(traj_st, traj_ed - self.seq_len + 1): # 遍历每个起始帧 st
                    ed = st + self.seq_len # 计算观察序列的终止帧 ed（一次性观察之前的10帧）
                    act_ed = st + self.seq_len + self.act_len - 1 # 计算动作序列的终止帧 act_ed（之后的5帧）
                    self.seq_tuple.append([traj_idx, st, ed, act_ed, traj_st, traj_ed,text]) # 将轨迹的索引、起始帧、终止帧、动作终止帧、轨迹起始帧、轨迹终止帧加入到序列元组中
                    #print(f"traj_idx: {traj_idx}, st: {st}, ed: {ed}, act_ed: {act_ed}, traj_st: {traj_st}, traj_ed: {traj_ed}, text: {text}")
        
        if self.use_play:
            # if you want to train with play data, please first generate play.json, and replace the following path
            with open(os.path.join(self.dataset_dir, "play.json"), 'r') as file:
                self.play_traj = json.load(file)
            n_trajs = len(self.play_traj['st_ed_list'])
            for traj_idx in range(n_trajs):
                traj_st, traj_ed = self.play_traj['st_ed_list'][traj_idx]
                if (traj_ed - self.seq_len - self.act_len + 2) <= traj_st:
                    continue
                # the action in the last frame is discarded
                for st in range(traj_st, traj_ed - self.seq_len + 1):
                    ed = st + self.seq_len
                    act_ed = st + self.seq_len + self.act_len - 1
                    self.seq_tuple.append([traj_idx, st, ed, act_ed, traj_st, traj_ed,''])
    
    def __len__(self):
        return len(self.seq_tuple)

    def __getitem__(self, index):   
        curr_tuple = self.seq_tuple[index]
        traj_idx = curr_tuple[0]
        st = curr_tuple[1]
        ed = curr_tuple[2]
        act_ed = curr_tuple[3]
        traj_st = curr_tuple[4]
        traj_ed = curr_tuple[5]       
        text=curr_tuple[6]

        static_rgbs = []
        hand_rgbs = []
        states = []


        state_buffer = [np.zeros(self.state_dim) for _ in range(self.seq_len + self.act_len - 1)] # 14维*14
        state_valid = [0 for _ in range(self.seq_len + self.act_len - 1)] # 14维

        tlen = ed - st # sequence length (10)
        assert tlen == self.seq_len

        for i in range(st, act_ed): # 从起始帧到动作终止帧, 一共14帧
            if i <= traj_ed:
                frame = self.load(os.path.join(self.dataset_dir, f"episode{traj_idx}", f"frame_{i:04d}.npz"))
                if i <= traj_ed:
                    state_buffer[i - st] = frame['robot_obs'] # 机器人joint action
                    state_valid[i - st] = 1
            
            if i < ed: # 从起始帧到观测终止帧, 一共10帧
                states += [frame['robot_obs']]
                
                static_rgb = frame['rgb_static']
                static_rgb = Image.fromarray(static_rgb) #(200, 200, 3)
                static_rgb = T.ToTensor()(static_rgb.convert("RGB")) # torch.Size([3, 200, 200])
                static_rgbs.append(static_rgb) # picture at time i

                hand_rgb = frame['rgb_gripper']
                hand_rgb = Image.fromarray(hand_rgb)
                hand_rgb = T.ToTensor()(hand_rgb.convert("RGB"))
                hand_rgbs.append(hand_rgb)

        # we use relative action represented in current end effector coordinate
        action_buffer = []
        action_valid = []
        if act_ed <= traj_ed: # 把最后一帧的动作加入到动作序列中
            act_ed_frame = self.load(os.path.join(self.dataset_dir, f"episode{traj_idx}", f"frame_{act_ed:04d}.npz"))
            state_buffer += [act_ed_frame['robot_obs']]
            state_valid += [1]
        else:
            state_buffer += [np.zeros(self.action_dim)] # mask non-existent action
            state_valid += [0]
        assert len(state_buffer) == (self.seq_len + self.act_len) # 15
        assert len(state_valid) == (self.seq_len + self.act_len) # 15
        for k in range(0, self.seq_len + self.act_len - 1):
            
            if (state_valid[k] and state_valid[k+1]):
                temp_action = np.zeros(self.action_dim)
                temp_action[:14] = state_buffer[k+1][0:14]
                # 提取并二值化处理
                # binary_gripper_state_left = 1 if state_buffer[k+1][6] > 0.023 else 0
                # temp_action[6] = binary_gripper_state_left
                # binary_gripper_state_right = 1 if state_buffer[k+1][13] > 0.023 else 0
                # temp_action[13] = binary_gripper_state_right

                action_buffer += [temp_action]
                action_valid += [1]
            else:
                action_buffer += [np.zeros(self.action_dim)]
                action_valid += [0]
        
        assert len(action_buffer) == (self.seq_len + self.act_len - 1) # 14
        assert len(action_valid) == (self.seq_len + self.act_len - 1) # 14

        # Prepare goal Image 随机一个输入的RGB goal image
        goal_min_idx = min(traj_ed, ed + self.forward_n_min)
        goal_max_idx = min(traj_ed + 1, ed + self.forward_n_max)
        goal_ids = list(range(goal_min_idx, goal_max_idx))
        goal_idx = random.choice(goal_ids)
        assert (goal_idx >= ed) and (goal_idx <= traj_ed)
        goal_frame = self.load(os.path.join(self.dataset_dir, f"episode{traj_idx}", f"frame_{goal_idx:04d}.npz"))
        goal_rgb = goal_frame['rgb_static']
        goal_rgb = Image.fromarray(goal_rgb)
        goal_rgb = T.ToTensor()(goal_rgb.convert("RGB"))
        static_rgbs.append(goal_rgb) # put goal_rgb at the end of static_rgbs

        # preprocess all Images
        static_rgbs = torch.stack(static_rgbs, dim=0)
        hand_rgbs = torch.stack(hand_rgbs, dim=0)
        if self.mode == 'train':
            static_rgbs = self.static_rgb_preprocess_train(static_rgbs)
            hand_rgbs = self.hand_rgb_preprocess_train(hand_rgbs)
        else:
            static_rgbs = self.static_rgb_preprocess_val(static_rgbs)
            hand_rgbs = self.static_rgb_preprocess_val(hand_rgbs)
        goal_rgb = static_rgbs[-1]
        static_rgbs = static_rgbs[:-1]

        # transform them into torch tensor
        _, C, H, W = static_rgbs.shape
        rgb_data  = torch.zeros((self.seq_len, C, H, W)).float() # (len, C, H, W)
        _, C, H, W = hand_rgbs.shape
        hand_rgb_data = torch.zeros((self.seq_len, C, H, W)).float() # (len, C, H, W)
        rgb_data [:tlen] = static_rgbs
        hand_rgb_data[:tlen] = hand_rgbs
        goal_rgb_data = goal_rgb.float() # (C, H, W)
    
        # State
        states = np.array(states)
        # left_gripper_state = states[:, 6]  # (10, ) 第6位是左臂的gripper状态
        # right_gripper_state = states[:, 13]  # (10, ) 第13位是右臂的gripper状态

        # #这里把gripper_states二值化
        # gripper_states_binary = np.where(left_gripper_state > 0.023, 1, 0)
        # states[:, 6] = gripper_states_binary
        # gripper_states_binary = np.where(right_gripper_state > 0.023, 1, 0)
        # states[:, 13] = gripper_states_binary
        states = torch.from_numpy(states)
        state_data = torch.zeros(self.seq_len, self.state_dim).float() # (len, state_dim)
        state_data[:tlen] = states

        # # Relative state
        # rel_states = np.zeros((tlen, self.state_dim), dtype=np.float32)
        # first_xyz = arm_states[0, 0:3]
        # first_rpy = arm_states[0, 3:6]
        # first_rotm = euler2rotm(first_rpy)
        # first_gripper = gripper_states[0]
        # rel_states[0, -1] = first_gripper
        # for i in range(1, tlen):
        #     curr_xyz = arm_states[i, 0:3]
        #     curr_rpy = arm_states[i, 3:6]
        #     curr_rotm = euler2rotm(curr_rpy)
        #     curr_gripper = gripper_states[i]
        #     rel_rotm = first_rotm.T @ curr_rotm
        #     rel_rpy = rotm2euler(rel_rotm)
        #     rel_xyz = first_rotm.T @ (curr_xyz - first_xyz)
        #     rel_states[i, 0:3] = rel_xyz
        #     rel_states[i, 3:6] = rel_rpy
        #     rel_states[i, 6] = curr_gripper
        # rel_state_data= torch.zeros(self.seq_len, self.state_dim).float() # (len, state_dim)
        # rel_state_data[:tlen] = torch.from_numpy(rel_states)

        # Action and action mask ，15帧的动作序列的掩码，5帧为一组
        actions = []
        action_masks = []
        for i in range(0, ed - st):
            actions.append(action_buffer[i:(i+self.act_len)])
            action_masks.append(action_valid[i:(i+self.act_len)])
        actions = np.array(actions) # (len, act_len, act_dim)
        actions = torch.from_numpy(actions)
        action_data  = torch.zeros(self.seq_len, self.act_len, self.action_dim).float() # (len, act_len, action_dim)
        action_data [:tlen] = actions
        action_masks = np.array(action_masks) # (len, act_len, act_dim)
        action_masks = torch.from_numpy(action_masks).long()
        action_mask_data = torch.zeros(self.seq_len, self.act_len).long() # (len, act_len, action_dim)
        action_mask_data[:tlen] = action_masks

        # Attention mask (should be all 1 for full dataset) 注意力掩码，全1
        attention_mask = np.ones(self.seq_len, dtype=np.int32) # (len)
        attention_mask[tlen:] = 0.0
        assert np.sum(attention_mask) == self.seq_len
        attention_mask_data = torch.from_numpy(attention_mask).long()


        #progress
        progress_data=torch.zeros(self.seq_len).float()
        for i in range(self.seq_len):
            progress=(st+i-traj_st)/(traj_ed-traj_st)
            progress_data[i]=progress

        
        data = dict()
        data['goal_rgb'] = goal_rgb_data # (C, H, W) (3, 224, 224)
        data['rgb'] = rgb_data # (len, C, H, W) (10, 3, 224, 224)
        data['hand_rgb'] = hand_rgb_data # (len, C, H, W) (10, 3, 224, 224)
        data['state'] = state_data # (len, state_dim) (10, 7)在训练没有用到！
        data['rel_state'] = state_data # (len, state_dim) (10, 7) 这里直接用state_data也就是joint action代替
        data['action'] = action_data # (len, act_len, action_dim) (10, 5, 7)需要预测的动作序列
        data['attention_mask'] = attention_mask_data # (len,) (10)
        data['action_mask'] = action_mask_data # (len, act_len, action_dim) (10, 5)
        data['text'] = [text] # (1)
        data["progress"]=progress_data #(len) (10)
        return data
  

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_dir = ""
    subfolder = 'task_ABC_D'
    seq_len = 10
    act_len = 5
    DS0 = CalvinDataset_Policy(
        data_dir,
        seq_len,
        act_len,
        forward_n_max=25,
        mode='train',
        subfolder=subfolder,
        use_data_augmentation=False,
        use_play=False,
        task_num=10000)
    
    rgb_mean = np.array([0.485, 0.456, 0.406])
    rgb_std  = np.array([0.229, 0.224, 0.225])
    for i in range(0, len(DS0),100):
        data = DS0[i]
        text = data['text']
        goal_rgb = data['goal_rgb']
        rgb = data['rgb']
        hand_rgb = data['hand_rgb']
        rel_state = data['rel_state']
        action = data['action']
        print(f"{text}")
        fig, ax = plt.subplots(seq_len // 3 + 1, 3)
        for k in range(seq_len):
            temp_rgb = rgb[k].permute(1, 2, 0).numpy()
            temp_rgb = temp_rgb * rgb_std + rgb_mean
            temp_rgb = np.clip(temp_rgb, 0.0, 1.0)
            ax[k // 3, k % 3].imshow(temp_rgb)
        temp_rgb = goal_rgb.permute(1, 2, 0).numpy()
        temp_rgb = temp_rgb * rgb_std + rgb_mean
        temp_rgb = np.clip(temp_rgb, 0.0, 1.0)
        ax[seq_len // 3, 2].imshow(temp_rgb)
        plt.savefig("debug_goal_rgb.png", dpi=300)

        fig, ax = plt.subplots(seq_len // 3 + 1, 3)
        for k in range(seq_len):
            temp_rgb = hand_rgb[k].permute(1, 2, 0).numpy()
            temp_rgb = temp_rgb * rgb_std + rgb_mean
            temp_rgb = np.clip(temp_rgb, 0.0, 1.0)
            ax[k // 3, k % 3].imshow(temp_rgb)
        plt.savefig("debug_goal_hand_rgb.png", dpi=300)
        
        accumulated_action = np.zeros(7)
        # the following is to check whether the computation of relative actions is correct
        for k in range(seq_len-1):
            curr_action = action[k, 0].numpy()
            curr_xyz = curr_action[0:3] / 50.0
            curr_rpy = curr_action[3:6] / 20.0
            curr_rotm = euler2rotm(curr_rpy)
            curr_gripper = curr_action[-1]

            accumulated_xyz = accumulated_action[0:3]
            accumulated_rpy = accumulated_action[3:6]
            accumulated_rotm = euler2rotm(accumulated_rpy)

            accumulated_xyz += np.dot(accumulated_rotm, curr_xyz)
            accumulated_rotm = accumulated_rotm @ curr_rotm
            accumulated_rpy = rotm2euler(accumulated_rotm) 

            next_rel_state = rel_state[k+1].numpy()
            assert np.allclose(curr_gripper, next_rel_state[-1]), "gripper"
            assert np.allclose(accumulated_xyz, next_rel_state[0:3]), "xyz"
            assert np.allclose(accumulated_rpy, next_rel_state[3:6]), "rpy"
            accumulated_action[0:3] = accumulated_xyz
            accumulated_action[3:6] = accumulated_rpy
            accumulated_action[-1] = curr_gripper