# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# command:
# python run_seq_tc_sota_41.6_bak.py -k cpn_ft_h36m_dbb -f 243 -s 243 -cf 256 -l log/exp12_1_cs512_eval -c checkpoint/exp12_1_cs512_eval -lr 0.00004 -lrd 0.99 -b 1024 -e 256 -cs 512 -dep 8 -gpu 3,4

import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno
import math

from einops import rearrange, repeat
from copy import deepcopy

from common.camera import *
import collections
from common.model_cross import *

from common.loss import *
from common.generators import ChunkedGenerator_Seq, UnchunkedGenerator_Seq
from time import time
from common.utils import *
from common.logging import Logger
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

#cudnn.benchmark = True       
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# import ptvsd
# ptvsd.enable_attach(address = ('192.168.210.130', 5678))
# print("ptvsd start")
# ptvsd.wait_for_attach()
# print("start debuging")
# joints_errs = []
args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


if args.evaluate != '':
    description = "Evaluate!"
elif args.evaluate == '':
    
    description = "Train!"

# initial setting
TIMESTAMP = "{0:%Y%m%dT%H-%M-%S/}".format(datetime.now())
# tensorboard
if not args.nolog:
    writer = SummaryWriter(args.log+'_'+TIMESTAMP)
    writer.add_text('description', description)
    writer.add_text('command', 'python ' + ' '.join(sys.argv))
    # logging setting
    logfile = os.path.join(args.log+'_'+TIMESTAMP, 'logging.log')
    sys.stdout = Logger(logfile)
print(description)
print('python ' + ' '.join(sys.argv))
print("CUDA Device Count: ", torch.cuda.device_count())
print(args)

# if not assign checkpoint path, Save checkpoint file into log folder
if args.checkpoint=='':
    args.checkpoint = args.log+'_'+TIMESTAMP
try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

# yuchen
human_num = 17
club_num = args.club_num
total_num = human_num + club_num
# end

print('Loading dataset...')

# IMPORTANT: For 3D ground truth, always use the 'gt' file
# The keypoints parameter is only for which 2D detections to use
dataset_path_3d = 'golfswing/data_3d_' + args.dataset + '_gt.npz'

if args.dataset == 'h36m':
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path_3d)
elif args.dataset == 'golf':
    from common.golf_dataset import GolfDataset
    # Use 'gt' for 3D data structure, but camera setup depends on keypoints type
    dataset = GolfDataset(dataset_path_3d, args.keypoints)
    # If args.keypoints is actually a path, force a safe type for GolfDataset
    kp_for_dataset = args.keypoints
    if isinstance(kp_for_dataset, str) and (kp_for_dataset.endswith(".npz") or kp_for_dataset.endswith(".npy")):
        kp_for_dataset = "gt"  # or whatever your dataset expects

    dataset = GolfDataset(dataset_path_3d, kp_for_dataset)

elif args.dataset.startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path_3d)
elif args.dataset.startswith('custom'):
    from common.custom_dataset import CustomDataset
    dataset = CustomDataset('data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in sorted(dataset.subjects()):
    for action in sorted(dataset[subject].keys()):
        anim = dataset[subject][action]

        if 'positions' in anim:
            # Unwrap single-camera list if needed
            if isinstance(anim['positions'], list):
                anim['positions'] = anim['positions'][0]   # (T, J, 3)

            print(f"\nProcessing {subject}/{action}")
            print(f"  Original shape: {anim['positions'].shape}")
            print(f"  Original range: {np.min(anim['positions']):.2f} to {np.max(anim['positions']):.2f}")

            # Trim JOINTS dimension to total_num (keep all time frames)
            if anim['positions'].shape[1] > total_num:
                print(f"  Trimming joints from {anim['positions'].shape[1]} to {total_num}")
                anim['positions'] = anim['positions'][:, :total_num, :]

            positions_3d = []
            for cam_idx, cam in enumerate(anim['cameras']):
                # The loaded data is in mm (range 0-900), already in correct coordinate system
                pos_3d_mm = anim['positions'].copy()
                
                # Convert mm to meters
                pos_3d = anim['positions'].copy()

                if np.max(np.abs(pos_3d)) > 10:
                    pos_3d = pos_3d / 1000.0

                pos_3d_meters = pos_3d

                
                print(f"  Camera {cam_idx} after meter conversion: {np.min(pos_3d_meters):.4f} to {np.max(pos_3d_meters):.4f} m")

                # Make root-relative for training
                # Subtract root position from all non-root joints
                root_position = pos_3d_meters[:, :1, :].copy()
                pos_3d_relative = pos_3d_meters - root_position
                
                print(f"  Camera {cam_idx} after root-relative: {np.min(pos_3d_meters):.4f} to {np.max(pos_3d_meters):.4f} m")
                print(f"    Root joint at frame 0: {pos_3d_meters[0, 0]}")
                print(f"    Joint 1 at frame 0: {pos_3d_meters[0, 1]}")

                positions_3d.append(pos_3d_relative)

            anim['positions_3d'] = positions_3d
            print(f"  Stored positions_3d with {len(positions_3d)} camera(s)")

print('Loading 2D detections...')

# Decide the keypoints path:
# 1) If args.keypoints is a direct path to an existing .npy/.npz, use it.
# 2) Otherwise, fall back to the original convention: golfswing/data_2d_{dataset}_{keypoints}.npz
if (
    isinstance(args.keypoints, str)
    and (args.keypoints.endswith(".npz") or args.keypoints.endswith(".npy"))
    and os.path.exists(args.keypoints)
):
    keypoints_path = args.keypoints
else:
    keypoints_path = f'golfswing/data_2d_{args.dataset}_{args.keypoints}.npz'

print(f"Loading 2D detections from: {keypoints_path}")

# Load depending on extension
keypoints_file = None
keypoints_array = None
keypoints_path = os.path.expanduser(str(keypoints_path)).strip().strip('"').strip("'")

ext = os.path.splitext(keypoints_path)[1].lower()

if keypoints_path.endswith(".npz"):
    keypoints_file = np.load(keypoints_path, allow_pickle=True)

elif keypoints_path.endswith(".npy"):
    keypoints_array = np.load(keypoints_path, allow_pickle=True)  # (T, J, 2) or (T, J, 3)

    # Ensure (T, J, 2)
    if keypoints_array.ndim == 4:
        keypoints_array = keypoints_array[0]
    if keypoints_array.shape[-1] == 3:
        keypoints_array = keypoints_array[..., :2]

    # Wrap into the same dict format as NPZ
    keypoints = {"custom": {"sequence": [keypoints_array]}}
    keypoints_metadata = {
        "layout_name": "custom_npy",
        "num_joints": keypoints_array.shape[1],
        "keypoints_symmetry": [
            list(range(0, keypoints_array.shape[1] // 2)),
            list(range(keypoints_array.shape[1] // 2, keypoints_array.shape[1]))
        ]
    }

else:
    raise ValueError(f"Unsupported keypoints file type: {keypoints_path}")
    

# Check what keys are available
available_keys = list(keypoints_file.keys())
print(f"Available keys in 2D keypoints file: {available_keys}")

# Try to load metadata from the detected file first, then fall back to GT file
keypoints_metadata = None

if 'metadata' in keypoints_file:
    keypoints_metadata = keypoints_file['metadata'].item()
    print("Using metadata from detected keypoints file")
else:
    print("WARNING: No metadata in detected keypoints file. Trying GT file...")
    # Try to load metadata from GT file
    try:
        gt_keypoints_file = np.load('golfswing/data_2d_' + args.dataset + '_gt.npz', allow_pickle=True)
        if 'metadata' in gt_keypoints_file:
            keypoints_metadata = gt_keypoints_file['metadata'].item()
            print("Using metadata from GT keypoints file")
        gt_keypoints_file.close()
    except:
        pass

# If still no metadata, create default
if keypoints_metadata is None:
    print("WARNING: No metadata found. Creating default metadata for 22 joints.")
    keypoints_metadata = {
        'layout_name': 'golf',
        'num_joints': 22,
        'keypoints_symmetry': [
            [1, 3, 5, 7, 9, 11],   # left joints (based on GT file)
            [2, 4, 6, 8, 10, 12]   # right joints (based on GT file)
        ]
    }

print(f"Metadata: {keypoints_metadata}")

# Extract symmetry info
keypoints_symmetry = keypoints_metadata.get('keypoints_symmetry', [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12]])
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

print(f"Keypoint symmetry - Left: {kps_left}, Right: {kps_right}")
print(f"Joint symmetry - Left: {joints_left}, Right: {joints_right}")

# Load positions_2d
if 'positions_2d' in keypoints_file:
    keypoints = keypoints_file['positions_2d'].item()
elif 'positions' in keypoints_file:
    keypoints = keypoints_file['positions'].item()
else:
    raise KeyError(f"Could not find 2D positions in keypoints file. Available keys: {available_keys}")

# Remap Step2 single-sequence NPZ -> viz subject/action (render mode only)
if args.render and isinstance(keypoints, dict) and "custom" in keypoints and "sequence" in keypoints["custom"]:
    seq = keypoints["custom"]["sequence"]

    # seq might be [array] or array
    arr = seq[0] if isinstance(seq, list) else seq  # (T, J, 2)

    # If accidentally saved (T, J, 3), drop confidence
    if arr.shape[-1] == 3:
        arr = arr[..., :2]
    
    
    if arr.shape[1] == 17:
        T = arr.shape[0]
        pad = np.zeros((T, 5, 2), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=1)   # (T, 22, 2)
        print("[INFO] Padded 2D keypoints 17 -> 22 (added 5 club joints as zeros)")

    import numpy as np

    print("\n[DEBUG] Joint y-values at frame 0 (sorted top->bottom):")
    ys = arr[0, :, 1]
    order = np.argsort(ys)  # small y = higher on image
    for rank, j in enumerate(order):
        x, y = arr[0, j]
        print(f"  rank {rank:02d}: j{j:02d}  x={x:7.2f}  y={y:7.2f}")

    # Print a quick guess: bottom-most joints (likely ankles/feet)
    bottom = order[-6:]
    print("\n[DEBUG] Bottom 6 joints (likely legs/feet):", bottom.tolist())
    for j in bottom:
        x, y = arr[0, j]
        print(f"  j{j:02d}: x={x:.2f}, y={y:.2f}")

    # Optional: labeled plot (frame 0)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(arr[0, :, 0], arr[0, :, 1])
        for j in range(arr.shape[1]):
            plt.text(arr[0, j, 0], arr[0, j, 1], str(j), fontsize=8)
        plt.gca().invert_yaxis()  # image coordinates
        plt.title("Frame 0 joint indices")
        plt.show()
    except Exception as e:
        print("[DEBUG] matplotlib plot skipped:", e)

    keypoints = {args.viz_subject: {args.viz_action: [arr]}}
    print(f"[INFO] Remapped keypoints custom/sequence -> {args.viz_subject}/{args.viz_action}")

print(f"Loaded 2D keypoints for subjects: {list(keypoints.keys())}")


# Check the structure and convert if needed
for subject in keypoints.keys():
    print(f"Subject {subject} actions: {list(keypoints[subject].keys())}")
    for action in keypoints[subject].keys():
        action_data = keypoints[subject][action]
        print(f"  Action {action}: type={type(action_data)}, shape={action_data.shape if hasattr(action_data, 'shape') else 'N/A'}")
        
        # If it's a single array (T, J, 2), wrap it in a list for camera dimension
        if isinstance(action_data, np.ndarray) and action_data.ndim == 3:
            print(f"    Converting to list format for camera compatibility")
            keypoints[subject][action] = [action_data]

# yuchen - remove 2D club kps if no club (trim to total_num joints)
for subject in sorted(keypoints.keys()):
    for action in sorted(keypoints[subject]):
        if isinstance(keypoints[subject][action], list):
            for cam_idx in range(len(keypoints[subject][action])):
                original_shape = keypoints[subject][action][cam_idx].shape
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][..., :total_num, :]
                new_shape = keypoints[subject][action][cam_idx].shape
                print(f"  Trimmed {subject}/{action}/cam{cam_idx}: {original_shape} -> {new_shape}")
        else:
            original_shape = keypoints[subject][action].shape
            keypoints[subject][action] = keypoints[subject][action][..., :total_num, :]
            new_shape = keypoints[subject][action].shape
            print(f"  Trimmed {subject}/{action}: {original_shape} -> {new_shape}")

# Update metadata with actual number of joints being used
keypoints_metadata['num_joints'] = total_num
print(f"Using {total_num} joints ({human_num} human + {club_num} club)")
# end






subjects_train = args.subjects_train.split(',')
subjects_test = args.subjects_test.split(',') if not args.render else [args.viz_subject]
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')

subjects_to_check = sorted(set(subjects_train + subjects_test + subjects_semi))

if not args.render:
    for subject in subjects_to_check:
        assert subject in keypoints, f'Subject {subject} is missing from the 2D detections dataset'
        for action in sorted(dataset[subject].keys()):
            assert action in keypoints[subject], f'Action {action} of subject {subject} is missing from the 2D detections dataset'

            if 'positions_3d' not in dataset[subject][action]:
                continue

            # ---------- FORCE SINGLE CAMERA ----------
            if isinstance(keypoints[subject][action], np.ndarray):
                keypoints[subject][action] = [keypoints[subject][action]]

            if isinstance(dataset[subject][action]['positions_3d'], np.ndarray):
                dataset[subject][action]['positions_3d'] = [dataset[subject][action]['positions_3d']]
            # -----------------------------------------

            num_cams = min(
                len(keypoints[subject][action]),
                len(dataset[subject][action]['positions_3d'])
            )

            for cam_idx in range(num_cams):
                kp = keypoints[subject][action][cam_idx]
                pos3d = dataset[subject][action]['positions_3d'][cam_idx]

                T = min(kp.shape[0], pos3d.shape[0])
                keypoints[subject][action][cam_idx] = kp[:T]
                dataset[subject][action]['positions_3d'][cam_idx] = pos3d[:T]
                print(
                f"[DEBUG] 2D range before model ({subject}/{action}/cam{cam_idx}): "
                f"{kp.min():.4f} → {kp.max():.4f}"
)
else:
    # Render mode: validate only the requested sequence exists
    s = args.viz_subject
    a = args.viz_action
    assert s in keypoints, f'Render mode: Subject {s} missing from 2D keypoints'
    assert a in keypoints[s], f'Render mode: Action {a} missing for subject {s}'

    if isinstance(keypoints[s][a], np.ndarray):
        keypoints[s][a] = [keypoints[s][a]]

    keypoints[s][a] = keypoints[s][a][:1]

    if 'positions_3d' in dataset[s][a]:
        if isinstance(dataset[s][a]['positions_3d'], np.ndarray):
            dataset[s][a]['positions_3d'] = [dataset[s][a]['positions_3d']]
        dataset[s][a]['positions_3d'] = dataset[s][a]['positions_3d'][:1]

        # Clamp lengths
        kp = keypoints[s][a][0]
        pos3d = dataset[s][a]['positions_3d'][0]
        T = min(kp.shape[0], pos3d.shape[0])
        keypoints[s][a][0] = kp[:T]
        dataset[s][a]['positions_3d'][0] = pos3d[:T]



from common.camera import normalize_screen_coordinates

def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    """
    Single-camera inference version of fetch()
    Assumptions:
    - One subject
    - One action
    - One camera
    - poses_2d shape: (T, J, 2)
    """

    out_poses_2d = []
    out_poses_3d = []
    out_camera_params = []

    for subject in sorted(subjects):
        assert subject in keypoints, f"Missing subject {subject} in 2D keypoints"

        for action in sorted(keypoints[subject].keys()):

            # Action filtering (kept for compatibility)
            if action_filter is not None:
                if not any(action.startswith(a) for a in action_filter):
                    continue

            # -----------------------------
            # 2D poses (ONE camera only)
            # -----------------------------
            poses_2d = np.asarray(keypoints[subject][action][0]).astype(np.float32)  # (T, J, 2)

            # If your input file has only 17 joints but model expects 22, pad here
            if poses_2d.shape[1] == 17 and args.club_num == 5:
                T = poses_2d.shape[0]
                pad = np.zeros((T, 5, 2), dtype=poses_2d.dtype)
                poses_2d = np.concatenate([poses_2d, pad], axis=1)  # (T, 22, 2)

            # Normalize to model expected coordinates (usually [-1, 1])
            # Only do this if it looks like pixel coordinates.
            cam0 = dataset.cameras()[subject][0]  # single cam
            # heuristic: pixels typically > 2; normalized usually within [-2, 2]
            if np.nanmax(np.abs(poses_2d)) > 2.5:
                poses_2d = normalize_screen_coordinates(
                    poses_2d, w=cam0['res_w'], h=cam0['res_h']
                )

            out_poses_2d.append(poses_2d)
            out_camera_params.append(None)

            # -----------------------------
            # 3D poses (optional / dummy)
            # -----------------------------
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                if isinstance(poses_3d, list):
                    poses_3d = poses_3d[0]
                out_poses_3d.append(poses_3d)

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
                

    return out_camera_params, out_poses_3d, out_poses_2d


action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

# set receptive_field as number assigned
receptive_field = args.number_of_frames
print('INFO: Receptive field: {} frames'.format(receptive_field))
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Receptive field', str(receptive_field))
pad = (receptive_field -1) // 2 # Padding on each side
min_loss = args.min_loss
width = cam['res_w']
height = cam['res_h']
num_joints = keypoints_metadata['num_joints']

#########################################PoseTransformer
model_pos_train =  MixSTE2(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0.1)

model_pos =  MixSTE2(num_frame=receptive_field, num_joints=num_joints, in_chans=2, embed_dim_ratio=args.cs, depth=args.dep,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,drop_path_rate=0)


################ load weight ########################
# posetrans_checkpoint = torch.load('./checkpoint/pretrained_posetrans.bin', map_location=lambda storage, loc: storage)
# posetrans_checkpoint = posetrans_checkpoint["model_pos"]
# model_pos_train = load_pretrained_weights(model_pos_train, posetrans_checkpoint)

#################
causal_shift = 0
model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params/1000000, 'Million')
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Trainable parameter count', str(model_params/1000000) + ' Million')

# make model parallel
if torch.cuda.is_available():
    model_pos = nn.DataParallel(model_pos)
    model_pos = model_pos.cuda()
    model_pos_train = nn.DataParallel(model_pos_train)
    model_pos_train = model_pos_train.cuda()

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    # chk_filename = args.resume or args.evaluate
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))

    model_pos_train_dict = model_pos_train.state_dict()
    model_pos_dict = model_pos.state_dict()
    if args.resume and args.resume.startswith('official_pretrained') and club_num != 0:
        checkpoint['model_pos'] = {k: v for k, v in checkpoint['model_pos'].items() if k in model_pos_train_dict and 'module.Spatial_pos_embed' not in k }
    model_pos_train_dict.update(checkpoint['model_pos'])
    model_pos_dict.update(checkpoint['model_pos'])

    model_pos_train.load_state_dict(model_pos_train_dict, strict=False)
    model_pos.load_state_dict(model_pos_dict, strict=False)


test_generator = UnchunkedGenerator_Seq(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
print('INFO: Testing on {} frames'.format(test_generator.num_frames()))
if not args.nolog:
    writer.add_text(args.log+'_'+TIMESTAMP + '/Testing Frames', str(test_generator.num_frames()))

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    # inputs_2d_p = torch.squeeze(inputs_2d)
    # inputs_3d_p = inputs_3d.permute(1,0,2,3)
    # out_num = inputs_2d_p.shape[0] - receptive_field + 1
    # eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    # for i in range(out_num):
    #     eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    # return eval_input_2d, inputs_3d_p
    ### split into (f/f1, f1, n, 2)
    assert inputs_2d.shape[:-1] == inputs_3d.shape[:-1], "2d and 3d inputs shape must be same! "+str(inputs_2d.shape)+str(inputs_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = torch.squeeze(inputs_3d)

    if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field: 
        out_num = inputs_2d_p.shape[0] // receptive_field+1
    elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field

    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_input_3d = torch.empty(out_num, receptive_field, inputs_3d_p.shape[1], inputs_3d_p.shape[2])

    for i in range(out_num-1):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        eval_input_3d[i,:,:,:] = inputs_3d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
    if inputs_2d_p.shape[0] < receptive_field:
        from torch.nn import functional as F
        pad_right = receptive_field-inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
        inputs_2d_p = F.pad(inputs_2d_p, (0,pad_right), mode='replicate')
        # inputs_2d_p = np.pad(inputs_2d_p, ((0, receptive_field-inputs_2d_p.shape[0]), (0, 0), (0, 0)), 'edge')
        inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
    if inputs_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field-inputs_3d_p.shape[0]
        inputs_3d_p = rearrange(inputs_3d_p, 'b f c -> f c b')
        inputs_3d_p = F.pad(inputs_3d_p, (0,pad_right), mode='replicate')
        inputs_3d_p = rearrange(inputs_3d_p, 'f c b -> b f c')
    eval_input_2d[-1,:,:,:] = inputs_2d_p[-receptive_field:,:,:]
    eval_input_3d[-1,:,:,:] = inputs_3d_p[-receptive_field:,:,:]

    return eval_input_2d, eval_input_3d


###################

# Training start
if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate
    optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    losses_3d_valid_human = []
    losses_3d_valid_club = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    # get training data
    
    # train_generator = ChunkedGenerator_Seq(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
    #                                    pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
    #                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    train_generator = ChunkedGenerator_Seq(args.batch_size//args.stride, cameras_train, poses_train, poses_train_2d, args.number_of_frames,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    train_generator_eval = UnchunkedGenerator_Seq(cameras_train, poses_train, poses_train_2d,
                                              pad=pad, causal_shift=causal_shift, augment=False)
    print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
    if not args.nolog:
        writer.add_text(args.log+'_'+TIMESTAMP + '/Training Frames', str(train_generator_eval.num_frames()))

    if args.resume:
        epoch = checkpoint['epoch']
        if False: #'optimizer' in checkpoint and checkpoint['optimizer'] is not None and not (args.resume.startswith('official_pretrained') and club_num != 0):

            # if args.resume.startswith('official_pretrained') and club_num != 0:
            #     checkpoint['optimizer']['state'][0]['exp_avg']

            optimizer.load_state_dict(checkpoint['optimizer'])
            train_generator.set_random_state(checkpoint['random_state'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        if not args.coverlr:
            lr = checkpoint['lr']

    print('** Note: reported losses are averaged over all frames.')
    print('** The final evaluation will be carried out after the last training epoch.')

    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        epoch_loss_traj_train = 0
        epoch_loss_2d_train_unlabeled = 0
        N = 0
        N_semi = 0
        model_pos_train.train()

        # Just train 1 time, for quick debug
        notrain=False
        for cameras_train, batch_3d, batch_2d in train_generator.next_epoch():
            # if notrain:break
            # notrain=True
            if cameras_train is not None:
                cameras_train = torch.from_numpy(cameras_train.astype('float32'))
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                if cameras_train is not None:
                    cameras_train = cameras_train.cuda()
            inputs_traj = inputs_3d[:, :, :1].clone()
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Predict 3D poses
            predicted_3d_pos = model_pos_train(inputs_2d)

            # del inputs_2d
            # torch.cuda.empty_cache()
            ### weight mpjpe
            if args.dataset=='h36m':
                # # hrdet
                # w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()

                w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
            elif args.dataset == 'golf':
                weights = [1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4, 4, 4, 4, 4, 4]
                # weights = [1 for _ in range(22)] # same as mpjpe
                w_mpjpe = torch.tensor(weights[:total_num]).cuda()

            elif args.dataset=='humaneva15':
                w_mpjpe = torch.tensor([1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]).cuda()
            loss_3d_pos = weighted_mpjpe(predicted_3d_pos, inputs_3d, w_mpjpe)
            # loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            # loss_3d_pos = cl_wmpjpe_switch(predicted_3d_pos, inputs_3d, w_mpjpe, epoch, args.epochs, minw=0.1, maxw=2.0)
            # loss_3d_pos = cl_wmpjpe(predicted_3d_pos, inputs_3d, w_mpjpe, epoch, iter_num=args.epochs, switch_iter=10)

            # Temporal Consistency Loss
            dif_seq = predicted_3d_pos[:,1:,:,:] - predicted_3d_pos[:,:-1,:,:]
            weights_joints = torch.ones_like(dif_seq).cuda()
            weights_mul = w_mpjpe
            assert weights_mul.shape[0] == weights_joints.shape[-2]
            weights_joints = torch.mul(weights_joints.permute(0,1,3,2),weights_mul).permute(0,1,3,2)
            # weights_diff = 0.5
            # index = [1,1,1,1,2,2,2,2,1]
            # dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)), dim=-1)
            dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))
            # loss_diff = (weights_diff * dif_seq)

            # weights_diff = 2.0
            loss_diff = 0.5 * dif_seq + 2.0 * mean_velocity_error_train(predicted_3d_pos, inputs_3d, axis=1)
            # loss_diff = 2.0 * mean_velocity_error_train(predicted_3d_pos, inputs_3d, axis=1)
            # loss_diff = 2.0 * mean_velocity_error_train(predicted_3d_pos, inputs_3d, axis=1)
            
            # norm_loss = Norm_Loss(receptive_field, 12, num_joints)
            # norm_loss_2 = Norm_Loss(receptive_field, 24, num_joints)
            # norm_loss_3 = Norm_Loss(receptive_field, 8, num_joints)
            # loss_diff += 0.001 * (norm_loss(predicted_3d_pos, inputs_3d) + \
            #                     norm_loss_2(predicted_3d_pos, inputs_3d) + \
            #                     norm_loss_3(predicted_3d_pos, inputs_3d))

            ### bone length consistency loss
            loss_bone = bonelen_consistency_loss(args.dataset, args.dataset, predicted_3d_pos)
            # if club_num == 5:
            #     loss_clublen = clublen_consistency_loss(args.dataset, args.keypoints, predicted_3d_pos)

            ### sym penalty loss
            # loss_sym = sym_penalty(args.dataset, args.keypoints, predicted_3d_pos)

            # loss_total = (loss_3d_pos[:,1:] + loss_diff)
            loss_total = loss_3d_pos + loss_diff + loss_bone

            # if club_num == 5:
            #     loss_total += loss_clublen
            
            loss_total.backward(loss_total.clone().detach())

            loss_total = torch.mean(loss_total)

            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_total.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            optimizer.step()
            # del inputs_3d, loss_3d_pos, predicted_3d_pos
            # torch.cuda.empty_cache()

        losses_3d_train.append(epoch_loss_3d_train / N)
        # torch.cuda.empty_cache()

        # End-of-epoch evaluation
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict(), strict=False)
            model_pos.eval()

            epoch_loss_3d_valid = 0
            epoch_loss_traj_valid = 0
            epoch_loss_2d_valid = 0
            epoch_loss_3d_vel = 0
            epoch_loss_3d_valid_human = 0
            epoch_loss_3d_valid_club = 0
            N = 0
            if not args.no_eval:
                # Evaluate on test set
                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                    ##### apply test-time-augmentation (following Videopose3d)
                    inputs_2d_flip = inputs_2d.clone()
                    inputs_2d_flip[:, :, :, 0] *= -1
                    inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]

                    ##### convert size
                    inputs_3d_p = inputs_3d
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                    inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                        inputs_2d_flip = inputs_2d_flip.cuda()
                    inputs_3d[:, :, 0] = 0

                    predicted_3d_pos = model_pos(inputs_2d)
                    predicted_3d_pos_flip = model_pos(inputs_2d_flip)
                    predicted_3d_pos_flip[:, :, :, 0] *= -1
                    predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                              joints_right + joints_left]
                    for i in range(predicted_3d_pos.shape[0]):
                        # print(predicted_3d_pos[i,0,0,0], predicted_3d_pos_flip[i,0,0,0])
                        predicted_3d_pos[i,:,:,:] = (predicted_3d_pos[i,:,:,:] + predicted_3d_pos_flip[i,:,:,:])/2
                        # print(predicted_3d_pos[i,0,0,0], predicted_3d_pos_flip[i,0,0,0])
                    # predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1, keepdim=True)

                    # del inputs_2d, inputs_2d_flip
                    # torch.cuda.empty_cache()

                    # set root as zero
                    # predicted_3d_pos[:, :, 0] = 0
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    if club_num != 0:
                        loss_3d_pos_human = mpjpe(predicted_3d_pos[..., :human_num, :], inputs_3d[..., :human_num, :])
                        loss_3d_pos_club = mpjpe(predicted_3d_pos[..., human_num:, :], inputs_3d[..., human_num:, :])
                        epoch_loss_3d_valid_human += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos_human.item()
                        epoch_loss_3d_valid_club += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos_club.item()

                    loss_3d_vel = mean_velocity_error_train(predicted_3d_pos, inputs_3d, axis=1)
                    epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_vel.item()


                    # del inputs_3d, loss_3d_pos, predicted_3d_pos
                    # torch.cuda.empty_cache()

                losses_3d_valid.append(epoch_loss_3d_valid / N)
                epoch_loss_3d_vel = epoch_loss_3d_vel/N

                losses_3d_valid_human.append(epoch_loss_3d_valid_human / N)
                losses_3d_valid_club.append(epoch_loss_3d_valid_club / N)

                # Evaluate on training set, this time in evaluation mode
                epoch_loss_3d_train_eval = 0
                epoch_loss_traj_train_eval = 0
                epoch_loss_2d_train_labeled_eval = 0
                N = 0
                for cam, batch, batch_2d in train_generator_eval.next_epoch():
                    if batch_2d.shape[1] == 0:
                        # This can only happen when downsampling the dataset
                        continue

                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)

                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()

                    inputs_3d[:, :, 0] = 0

                    # Compute 3D poses
                    predicted_3d_pos = model_pos(inputs_2d)

                    # del inputs_2d
                    # torch.cuda.empty_cache()
                    
                    # set root as zero
                    # predicted_3d_pos[:, :, 0] = 0
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_train_eval += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    # del inputs_3d, loss_3d_pos, predicted_3d_pos
                    # torch.cuda.empty_cache()

                losses_3d_train_eval.append(epoch_loss_3d_train_eval / N)

                # Evaluate 2D loss on unlabeled training set (in evaluation mode)
                epoch_loss_2d_train_unlabeled_eval = 0
                N_semi = 0

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_eval %f 3d_valid %f 3d_val_velocity %f   3d_valid_human %f   3d_valid_club %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses_3d_train[-1] * 1000,
                losses_3d_train_eval[-1] * 1000,
                losses_3d_valid[-1] * 1000,
                epoch_loss_3d_vel * 1000,
                losses_3d_valid_human[-1] * 1000,
                losses_3d_valid_club[-1] * 1000
                ))
            if not args.nolog:
                writer.add_scalar("Loss/3d training eval loss", losses_3d_train_eval[-1] * 1000, epoch+1)
                writer.add_scalar("Loss/3d validation loss", losses_3d_valid[-1] * 1000, epoch+1)
        if not args.nolog:
            writer.add_scalar("Loss/3d training loss", losses_3d_train[-1] * 1000, epoch+1)
            writer.add_scalar("Parameters/learing rate", lr, epoch+1)
            writer.add_scalar('Parameters/training time per epoch', elapsed, epoch+1)
        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        # momentum = initial_momentum * np.exp(-epoch/args.epochs * np.log(initial_momentum/final_momentum))
        # model_pos_train.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'min_loss': min_loss
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, chk_path)

        #### save best checkpoint
        best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin'.format(epoch))
        # min_loss = 41.65
        if losses_3d_valid[-1] * 1000 < min_loss:
            min_loss = losses_3d_valid[-1] * 1000
            print("save best checkpoint")
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                # 'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                # 'random_state_semi': semi_generator.random_state() if semi_supervised else None,
            }, best_chk_path)

        # Save training curves after every epoch, as .png images (if requested)
        if args.export_training_curves and epoch > 3:
            if 'matplotlib' not in sys.modules:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

            plt.figure()
            epoch_x = np.arange(3, len(losses_3d_train)) + 1
            plt.plot(epoch_x, losses_3d_train[3:], '--', color='C0')
            plt.plot(epoch_x, losses_3d_train_eval[3:], color='C0')
            plt.plot(epoch_x, losses_3d_valid[3:], color='C1')
            plt.legend(['3d train', '3d train (eval)', '3d valid (eval)'])
            plt.ylabel('MPJPE (m)')
            plt.xlabel('Epoch')
            plt.xlim((3, epoch))
            plt.savefig(os.path.join(args.checkpoint, 'loss_3d.png'))

            plt.close('all')
# Training end

# Evaluate
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False, newmodel=None):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    epoch_loss_3d_pos_human = 0
    epoch_loss_3d_pos_club = 0
    with torch.no_grad():
        if newmodel is not None:
            print('Loading comparison model')
            model_eval = newmodel
            chk_file_path = '/mnt/data3/home/zjl/workspace/3dpose/PoseFormer/checkpoint/train_pf_00/epoch_60.bin'
            print('Loading evaluate checkpoint of comparison model', chk_file_path)
            checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
            model_eval.load_state_dict(checkpoint['model_pos'], strict=False)
            model_eval.eval()
        else:
            model_eval = model_pos
            if not use_trajectory_model:
                # load best checkpoint
                if args.evaluate == '':
                    chk_file_path = os.path.join(args.checkpoint, 'best_epoch.bin')
                    print('Loading best checkpoint', chk_file_path)
                elif args.evaluate != '':
                    chk_file_path = os.path.join(args.checkpoint, args.evaluate)
                    print('Loading evaluate checkpoint', chk_file_path)
                checkpoint = torch.load(chk_file_path, map_location=lambda storage, loc: storage)
                print('This model was trained for {} epochs'.format(checkpoint['epoch']))
                # model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
                model_eval.load_state_dict(checkpoint['model_pos'], strict=False)
                model_eval.eval()
        # else:
            # model_traj.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))


            ##### apply test-time-augmentation (following Videopose3d)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            ##### convert size
            inputs_3d_p = inputs_3d
            if newmodel is not None:
                def eval_data_prepare_pf(receptive_field, inputs_2d, inputs_3d):
                    inputs_2d_p = torch.squeeze(inputs_2d)
                    inputs_3d_p = inputs_3d.permute(1,0,2,3)
                    padding = int(receptive_field//2)
                    inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
                    inputs_2d_p = F.pad(inputs_2d_p, (padding,padding), mode='replicate')
                    inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
                    out_num = inputs_2d_p.shape[0] - receptive_field + 1
                    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
                    for i in range(out_num):
                        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
                    return eval_input_2d, inputs_3d_p
                
                inputs_2d, inputs_3d = eval_data_prepare_pf(81, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare_pf(81, inputs_2d_flip, inputs_3d_p)
            else:
                inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d_p)
                inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d_p)

            # if newmodel is not None:
            #     bi, ti, ni, _ = inputs_2d.shape
            #     inputs_2d = inputs_2d.reshape(int(bi*3), int(ti/3), ni, -1)
            #     inputs_3d = inputs_3d.reshape(int(bi*3), int(ti/3), ni, -1)
            #     inputs_2d_flip = inputs_2d_flip.reshape(int(bi*3), int(ti/3), ni, -1)

            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()
                inputs_2d_flip = inputs_2d_flip.cuda()
                inputs_3d = inputs_3d.cuda()
                
            inputs_3d[:, :, 0] = 0

            save_correlation = False
            if action == 'Swing05' and  batch.shape[1] == 971:
                save_correlation = True
            
            predicted_3d_pos = model_eval(inputs_2d, save_correlation)
            predicted_3d_pos_flip = model_eval(inputs_2d_flip)
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :,
                                                                      joints_right + joints_left]
            for i in range(predicted_3d_pos.shape[0]):
                predicted_3d_pos[i,:,:,:] = (predicted_3d_pos[i,:,:,:] + predicted_3d_pos_flip[i,:,:,:])/2
            # predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1, keepdim=True)

            # del inputs_2d, inputs_2d_flip
            # torch.cuda.empty_cache()

            if return_predictions:
                return predicted_3d_pos.squeeze().cpu().numpy()

            error = mpjpe(predicted_3d_pos, inputs_3d)

            if club_num != 0:
                error_human = mpjpe(predicted_3d_pos[..., :human_num, :], inputs_3d[..., :human_num, :])
                error_club = mpjpe(predicted_3d_pos[..., human_num:, :], inputs_3d[..., human_num:, :])

            # error, joints_err = mpjpe(predicted_3d_pos, inputs_3d, return_joints_err=True)
            # joints_errs.append(joints_err)

            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            if club_num != 0:
                epoch_loss_3d_pos_human += inputs_3d.shape[0]*inputs_3d.shape[1] * error_human.item()
                epoch_loss_3d_pos_club += inputs_3d.shape[0]*inputs_3d.shape[1] * error_club.item()

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----'+action+'----')
    e1 = (epoch_loss_3d_pos / N)*1000
    e2 = (epoch_loss_3d_pos_procrustes / N)*1000
    e3 = (epoch_loss_3d_pos_scale / N)*1000
    ev = (epoch_loss_3d_vel / N)*1000
    e1_human = (epoch_loss_3d_pos_human / N)*1000
    e1_club = (epoch_loss_3d_pos_club / N)*1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    if club_num != 0:
        print('Protocol #1 Human Error (MPJPE):', e1_human, 'mm')
        print('Protocol #1 Club Error (MPJPE):', e1_club, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e1_human, e1_club, e2, e3, ev


if args.render:
    print('Rendering...')

    input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
    ground_truth = None
    if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
        if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
            ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
    if ground_truth is None:
        print('INFO: this action is unlabeled. Ground truth will not be rendered.')

    print(f"Original data - Input: {input_keypoints.shape}, GT: {ground_truth.shape if ground_truth is not None else None}")
    
    gen = UnchunkedGenerator_Seq(None, [ground_truth], [input_keypoints],
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    
    print(f'INFO: Generator reports {gen.num_frames()} frames')
    
    # Get predictions
    prediction = evaluate(gen, return_predictions=True)
    
    print(f"Prediction shape after evaluate: {prediction.shape}")
    


# Step 2: Fix coordinate system

print("\n=== Fixing Coordinate System ===")

# Define transformation function
def transform_coordinates(data):
    """Transform coordinates: swap Y/Z and flip Z"""
    transformed = data.copy()
    # transformed[:, :, 0] = -data[:, :, 0]  # FIX: flip X
    # transformed[:, :, 1] = data[:, :, 2]   # Y = old Z
    # transformed[:, :, 2] = -data[:, :, 1]   # New Z = -old Y
    return transformed

# # Step 1: Flatten if batched
# if prediction.ndim == 4:
#     print(f"Flattening batched predictions from {prediction.shape}")
#     prediction = prediction.reshape(-1, prediction.shape[-2], prediction.shape[-1])
# print(f"Flattened to: {prediction.shape}")

# # Step 2: Transform prediction (root-relative)
# print("Transforming prediction coordinates...")
# prediction = transform_coordinates(prediction)

# # Step 3: Trim to match ground truth
# if ground_truth is not None:
#     target_len = ground_truth.shape[0]
#     print(f"\n=== Matching Lengths ===")
#     print(f"Prediction: {prediction.shape[0]} frames")
#     print(f"Ground truth: {target_len} frames")
    
#     if prediction.shape[0] > target_len:
#         prediction = prediction[:target_len]
#     elif prediction.shape[0] < target_len:
#         ground_truth = ground_truth[:prediction.shape[0]]

# Step 4: Load absolute ground truth and reconstruct absolute prediction
if args.viz_output is not None and ground_truth is not None:
    print("\n=== Preparing Visualization ===")
    
    dataset_path_3d = 'golfswing/data_3d_' + args.dataset + '_gt.npz'
    raw_data = np.load(dataset_path_3d, allow_pickle=True)
    
    if 'positions_3d' in raw_data:
        all_positions = raw_data['positions_3d'].item()
        original_positions = all_positions[args.viz_subject][args.viz_action]
        
        if isinstance(original_positions, list):
            original_positions = original_positions[args.viz_camera]
        
        # Trim to match
        original_positions = original_positions[:prediction.shape[0]]
        if original_positions.shape[1] > prediction.shape[1]:
            original_positions = original_positions[:, :prediction.shape[1], :]
        
        # Convert mm to meters if needed
        if np.max(np.abs(original_positions)) > 10:
            original_positions = original_positions / 1000.0
        
        # CRITICAL: Transform ground truth to match prediction coordinate system
        print("Transforming ground truth to match prediction coordinates...")
        original_positions = transform_coordinates(original_positions)
        
        # Extract trajectory (root joint positions)
        trajectory = original_positions[:, :1, :].copy()
        
        # Reconstruct absolute prediction (now in same coordinate system)
        prediction_absolute = prediction + trajectory
        
        print(f"After transformation:")
        print(f"  Prediction: range=[{np.min(prediction_absolute):.3f}, {np.max(prediction_absolute):.3f}]")
        print(f"  Ground truth: range=[{np.min(original_positions):.3f}, {np.max(original_positions):.3f}]")
        
        prediction_final = prediction_absolute
        ground_truth_final = original_positions
        

    else:
        prediction_final = prediction
        ground_truth_final = ground_truth
else:
    prediction_final = prediction
    ground_truth_final = None


# Step 4: Export if requested
if args.viz_export is not None:
    print('Exporting joint positions to', args.viz_export)
    np.save(args.viz_export, prediction)

# Step 5: Prepare for visualization
if args.viz_output is not None:
    print("\n=== Preparing Visualization ===")
    
    # Step 1: Flatten prediction if batched
    if prediction.ndim == 4:
        print(f"Flattening batched predictions from {prediction.shape}")
        prediction = prediction.reshape(-1, prediction.shape[-2], prediction.shape[-1])
    print(f"Prediction shape (root-relative): {prediction.shape}")
    
    if ground_truth is not None:
        # Step 2: Load ORIGINAL absolute positions (NO TRANSFORMATION)
        dataset_path_3d = 'golfswing/data_3d_' + args.dataset + '_gt.npz'
        print(f"\nLoading original absolute positions from: {dataset_path_3d}")
        
        try:
            raw_data = np.load(dataset_path_3d, allow_pickle=True)
            
            if 'positions_3d' in raw_data:
                all_positions = raw_data['positions_3d'].item()
                original_positions = all_positions[args.viz_subject][args.viz_action]
                
                if isinstance(original_positions, list):
                    original_positions = original_positions[args.viz_camera]
                
                # Trim to match prediction length
                original_positions = original_positions[:prediction.shape[0]]
                if original_positions.shape[1] > prediction.shape[1]:
                    original_positions = original_positions[:, :prediction.shape[1], :]
                
                # Convert mm to meters if needed
                if np.max(np.abs(original_positions)) > 10:
                    original_positions = original_positions / 1000.0
                
                print(f"Loaded absolute GT: shape={original_positions.shape}")
                print(f"  Range: [{np.min(original_positions):.3f}, {np.max(original_positions):.3f}]")
                
                # ============================================
                # SIMPLE APPROACH: Don't transform anything
                # Just extract trajectory and add to prediction
                # ============================================
                trajectory = original_positions[:, :1, :].copy()
                prediction_absolute = prediction + trajectory
                
                prediction_final = prediction_absolute
                ground_truth_final = original_positions
                
                print(f"\nFinal data (no transformation):")
                print(f"  Prediction: [{np.min(prediction_final):.3f}, {np.max(prediction_final):.3f}]")
                print(f"  Ground truth: [{np.min(ground_truth_final):.3f}, {np.max(ground_truth_final):.3f}]")
                print(f"  Prediction frame 0, joint 0: {prediction_final[0, 0]}")
                print(f"  Ground truth frame 0, joint 0: {ground_truth_final[0, 0]}")
                
            else:
                raise KeyError("'positions_3d' not found in raw data file")
                
        except Exception as e:
            print(f"ERROR loading raw data: {e}")
            prediction_final = prediction
            ground_truth_final = ground_truth
    else:
        prediction_final = prediction
        ground_truth_final = None
    
    # Create TWO separate animations to diagnose the issue
    # This will show ground truth and prediction side-by-side
    anim_output = {}
    
    # if ground_truth_final is not None:
    #     anim_output['Ground truth'] = ground_truth_final
    #     print("\nGround truth will be rendered")
    
    anim_output['Reconstruction'] = prediction_final
    print("Reconstruction will be rendered")
    
    # Render
    cam = dataset.cameras()[args.viz_subject][args.viz_camera]
    input_keypoints_viz = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
    
    from common.visualization import render_animation
    
    # ============================================
    # KEY INSIGHT: The issue is likely in the skeleton definition
    # or the camera azimuth angle, NOT the coordinate data
    # ============================================
    
    # Try different azimuth angles to see which one aligns them
    # Uncomment one at a time to test:
    
    azimuth_angle = cam['azimuth']          
    # azimuth_angle = cam['azimuth'] + 90   
    # azimuth_angle = cam['azimuth'] - 90   
    # azimuth_angle = cam['azimuth'] + 180  
    # azimuth_angle = 70                    
    
    print(f"\n=== Rendering with azimuth: {azimuth_angle} (original: {cam['azimuth']}) ===")
    
    
    render_animation(input_keypoints_viz, keypoints_metadata, anim_output,
    dataset.skeleton(), dataset.fps(), args.viz_bitrate, azimuth_angle, args.viz_output,
    limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
    input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
    input_video_skip=args.viz_skip)
    
    print(f"\n Successfully rendered {prediction_final.shape[0]} frames to {args.viz_output}")

        

else:
    print('Evaluating...')
    os.makedirs("vis_correlation", exist_ok=True)
    all_actions = {}
    all_actions_by_subject = {}
    for subject in sorted(subjects_test):
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}

        for action in sorted(dataset[subject].keys()):
            action_name = action.split(' ')[0]
            if action_name not in all_actions:
                all_actions[action_name] = []
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions[action_name].append((subject, action))
            all_actions_by_subject[subject][action_name].append((subject, action))

    def fetch_actions(actions):
        out_poses_3d = []
        out_poses_2d = []

        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)): # Iterate across cameras
                out_poses_3d.append(poses_3d[i])

        stride = args.downsample
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]

        return out_poses_3d, out_poses_2d

    def run_evaluation(actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []

        errors_p1_human = []
        errors_p1_club = []
        # joints_errs_list=[]

        for action_key in actions.keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_act, poses_2d_act = fetch_actions(actions[action_key])
            gen = UnchunkedGenerator_Seq(None, poses_act, poses_2d_act,
                                     pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                     kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                     joints_right=joints_right)
            e1, e1_human, e1_club, e2, e3, ev = evaluate(gen, action_key)
            
            # joints_errs_list.append(joints_errs)

            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)
            errors_p1_human.append(e1_human)
            errors_p1_club.append(e1_club)

        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #1   (Human)(MPJPE) action-wise average:', round(np.mean(errors_p1_human), 1), 'mm')
        print('Protocol #1   (Club)(MPJPE) action-wise average:', round(np.mean(errors_p1_club), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')

        # joints_errs_np = np.array(joints_errs_list).reshape(-1, 17)
        # joints_errs_np = np.mean(joints_errs_np, axis=0).reshape(-1)
        # with open('output/mpjpe_joints.csv', 'a+') as f:
        #     for i in joints_errs_np:
        #         f.write(str(i)+'\n')

    if not args.by_subject:
        run_evaluation(all_actions, action_filter)
    else:
        for subject in all_actions_by_subject.keys():
            print('Evaluating on subject', subject)
            run_evaluation(all_actions_by_subject[subject], action_filter)
            print('')
if not args.nolog:
    writer.close()

