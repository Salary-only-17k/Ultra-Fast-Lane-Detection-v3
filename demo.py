import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import tqdm
from data.dataset import LaneTestDataset
from utils.common import get_model, merge_config
from utils.dist_utils import dist_print

COLORS = [
    (128, 255, 0),  # 明绿
    (255, 128, 0),   # 明蓝
    (128, 0, 255),   # 粉色
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def label2str(pre_lane_classes):
    cls_id = {'0':'',
              '1':"white_solid",
              '2':"white_dashed",
              '3':"yellow_solid",
              '4':"yellow_dashed"}
    
    confs,lanes_cls =  torch.max(pre_lane_classes,dim=2)
    text = []   # label conf
    # print(f">>>-----------<<<")
    # print(pre_lane_classes.shape)
    # print(confs,lanes_cls)
    # print(f">>>{lanes_cls}<<<")
    for lane_cls,conf in zip(lanes_cls.numpy()[0],confs.numpy()[0]):
        text.append(f'label : {cls_id[str(lane_cls)]}   conf : {conf:.3f}')
    return text
        
            

def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
    _, num_grid_row, num_cls_row, _ = pred['loc_row'].shape
    _, num_grid_col, num_cls_col, _ = pred['loc_col'].shape
    batch_size, num_classes, num_lanes = pred['lane_labels'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    pre_lane_classes = pred['lane_labels'].cpu()
    infos = label2str(pre_lane_classes)
    
    coords = []

    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)
    # print('coords: ',coords)
    return coords,infos
if __name__ == "__main__":
    res= r'runs/test'
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()
    cfg.batch_size = 1
    print('setting batch_size to 1 for demo generation')

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = get_model(cfg)

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((int(cfg.train_height / cfg.crop_ratio), cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        # splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        # datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
        # img_w, img_h = 1640, 590
        splits = ['train_gt.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list',split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
        img_w, img_h = 1640, 590
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms, crop_size = cfg.train_height) for split in splits]
        img_w, img_h = 1280, 720
    else:
        raise NotImplementedError
    for split, dataset in zip(splits, datasets):   # [test_file_lst, loader_lst]
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
        
        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                pred = net(imgs)

            vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
            # print(vis.shape)
            coords,infos = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width = img_w, original_image_height = img_h)
            # print('coords: ',coords)
            if len(coords)==0:
                continue
            else:
                for coord in coords[0]:
                    # print('lane: ',lane)
                    # for coord in lane:
                    # print(vis.shape,vis.dtype)
                    # print('coord: ',coord)
                    try:
                        cv2.circle(vis,coord,5,(0,255,0),-1)
                    except:
                        continue
            for i,(info,colr) in enumerate(zip(infos,COLORS)):
                cv2.putText(vis,info, (10,int(40*(i+1))), cv2.FONT_HERSHEY_SIMPLEX, 0.8,colr, 2, cv2.LINE_AA)
            save_img_pth = os.path.join(res,names[0])
            os.makedirs(os.path.dirname(save_img_pth),exist_ok=True)
            cv2.imwrite(save_img_pth,vis)
           

