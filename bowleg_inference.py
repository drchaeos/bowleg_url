## 필요한 모듈 import

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import csv
import math

import torch
from typing import Tuple, List, Sequence, Callable, Dict

from detectron2.structures import BoxMode
from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog


## parameter 설정
dataname = 'bow_leg'
column_path = './columns/bow_leg.csv'
test_dir = './test_imgs'
output_dir = './out'
num_keypoints = 4
keypoint_names = {0: 'LK', 1: 'MK', 2: 'LA', 3: 'MA'}
edges = [(0, 1), (2, 3)]

## 함수 정의
def cal_bowleg(image, RtC1, RtC2, LtC1, LtC2):
    h, w, c = image.shape
    np.random.seed(42)
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range((num_keypoints) + 3)}
    D1 = abs(RtC2[0] - LtC2[0])
    D3 = abs(RtC1[0] - ((RtC2[0]+LtC2[0]) / 2))
    D2 = abs(LtC1[0] - ((RtC2[0]+LtC2[0]) / 2))

    if D1 == 0:
        Rt_PLEAA = "Undefined due to D1 being zero"
        Lt_PLEAA = "Undefined due to D1 being zero"
    else:
        Rt_PLEAA = format(-27.24 + (27.39 * (D3 / (D1 / 2))), ".3f")
        Lt_PLEAA = format(-27.24 + (27.39 * (D2 / (D1 / 2))), ".3f")

    if RtC1 and LtC1 and RtC2 and LtC2:
        cv2.line(image, (int(RtC1[0]), int(RtC1[1])), (int((RtC2[0]+LtC2[0]) / 2), int(RtC1[1])), (0,0,128), 3, lineType=cv2.LINE_8)
        cv2.line(image, (int(LtC1[0]), int(LtC1[1])), (int((RtC2[0]+LtC2[0]) / 2), int(LtC1[1])), (0,0,128), 3, lineType=cv2.LINE_8)
    
    if RtC2 and LtC2:
        cv2.line(image, (int(RtC2[0]), int(RtC2[1])), (int(LtC2[0]), int(RtC2[1])), (0,0,128), 3, lineType=cv2.LINE_8)
        cv2.line(image, (int((RtC2[0]+LtC2[0]) / 2), h), (int((RtC2[0]+LtC2[0]) / 2), 0), (0,0,0), 2, lineType=cv2.LINE_AA)

    return image, (Rt_PLEAA, Lt_PLEAA)


def reset(RtC1, RtC2, LtC1, LtC2, PLEAA):
    if RtC1:
        RtC1 = "NaN"
    if RtC2:
        RtC2 = "NaN"
    if LtC1:
        LtC1 = "NaN"
    if LtC2:
        LtC2 = "NaN"
    if PLEAA:
        PLEAA = "NaN"

    return RtC1, RtC2, LtC1, LtC2, PLEAA


def draw_keypoints(image, keypoints,
                   edges: List[Tuple[int, int]] = None,
                   keypoint_names: Dict[int, str] = None,
                   boxes: bool = True) -> None:

    keypoints = keypoints.astype(np.int64)
    keypoints_ = keypoints.copy()
    np.random.seed(42)
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range((num_keypoints) + 2)}
    if len(keypoints_) == (2 * num_keypoints):
        keypoints_ = [[keypoints_[i], keypoints_[i + 1]] for i in range(0, len(keypoints_), 2)]

    assert isinstance(image, np.ndarray), "image argument does not numpy array."
    image_ = np.copy(image)
    for i, keypoint in enumerate(keypoints_):
        cv2.circle(
            image_,
            tuple(keypoint),
            3, colors.get(i), thickness=3, lineType=cv2.FILLED)

        if keypoint_names is not None:
            cv2.putText(
                image_,
                f'{i}: {keypoint_names[i]}',
                tuple(keypoint),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if edges is not None:
        for i, edge in enumerate(edges):
            cv2.line(
                image_,
                tuple(keypoints_[edge[0]]),
                tuple(keypoints_[edge[1]]),
                colors.get(edge[0]), 5, lineType=cv2.LINE_AA)
    if boxes:
        x1, y1 = min(np.array(keypoints_)[:, 0]), min(np.array(keypoints_)[:, 1])
        x2, y2 = max(np.array(keypoints_)[:, 0]), max(np.array(keypoints_)[:, 1])
        cv2.rectangle(image_, (x1, y1), (x2, y2), (255, 100, 91), thickness=3)

    h, w, c = image.shape

    # 모든 점 x,y 좌표 추출
    LK_x = keypoints_[0][0]
    LK_y = keypoints_[0][1]
    MK_x = keypoints_[1][0]
    MK_y = keypoints_[1][1]
    LA_x = keypoints_[2][0]
    LA_y = keypoints_[2][1]
    MA_x = keypoints_[3][0]
    MA_y = keypoints_[3][1]

    global RtC1, RtC2, LtC1, LtC2
    if LK_x < (w / 2):
        RtC1 = (((LK_x + MK_x) / 2), ((LK_y + MK_y) / 2))
        RtC2 = (((LA_x + MA_x) / 2), ((LA_y + MA_y) / 2))
        cv2.circle(image_, (int(RtC1[0]), int(RtC1[1])), 3, colors.get(5), thickness=3, lineType=cv2.FILLED)
        cv2.circle(image_, (int(RtC2[0]), int(RtC2[1])), 3, colors.get(6), thickness=3, lineType=cv2.FILLED)
        cv2.putText(image_, "C1", (int(RtC1[0]), int(RtC1[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image_, "C2", (int(RtC2[0]), int(RtC2[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    if LK_x > (w / 2):
        LtC1 = (((LK_x + MK_x) / 2), ((LK_y + MK_y) / 2))
        LtC2 = (((LA_x + MA_x) / 2), ((LA_y + MA_y) / 2))
        cv2.circle(image_, (int(LtC1[0]), int(LtC1[1])), 3, colors.get(5), thickness=3, lineType=cv2.FILLED)
        cv2.circle(image_, (int(LtC2[0]), int(LtC2[1])), 3, colors.get(6), thickness=3, lineType=cv2.FILLED)
        cv2.putText(image_, "C1", (int(LtC1[0]), int(LtC1[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image_, "C2", (int(LtC2[0]), int(LtC2[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return image_


def save_samples(dst_path, image_path, csv_path, mode="random", size=None, index=None):
    keypoint_names = {0: 'LK', 1: 'MK', 2: 'LA', 3: 'MA'}
    edges = [(0, 1), (2, 3)]
    df = pd.read_csv(csv_path)

    # csv 파일로 저장
    output_file = open(os.path.join(dst_path, "PLEAA_result.csv"), 'w', newline='')
    f = csv.writer(output_file)
    # csv 파일에 header 추가
    f.writerow(["image name", "우측", "좌측"])

    if mode == "random":
        assert size is not None, "mode argument is random, but size argument is not given."
        choice_idx = np.random.choice(len(df), size=size, replace=False)
    if mode == "choice":
        assert index is not None, "mode argument is choice, but index argument is not given."
        choice_idx = index

    global RtC1, RtC2, LtC1, LtC2, PLEAA

    for idx in choice_idx:
        image_name = df.iloc[idx, 0]
        keypoints = df.iloc[idx, 1:]
        image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_COLOR)
        if idx == 0:
            combined = draw_keypoints(image, keypoints, edges, keypoint_names, boxes=False)
            if image_name != df.iloc[idx + 1, 0]:
                combined, PLEAA = cal_bowleg(combined, RtC1, RtC2, LtC1, LtC2)
                cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
                f.writerow([image_name, str(int(float(PLEAA[0]))), str(int(float(PLEAA[1])))])
                reset(RtC1, RtC2, LtC1, LtC2, PLEAA)

        if 0 < idx < (len(choice_idx)-1):
            if image_name == df.iloc[idx + 1, 0]:
                if image_name != df.iloc[idx - 1, 0]:
                    combined = draw_keypoints(image, keypoints, edges, keypoint_names, boxes=False)
                if image_name == df.iloc[idx - 1, 0]:
                    combined = draw_keypoints(combined, keypoints, edges, keypoint_names, boxes=False)

            if image_name != df.iloc[idx + 1, 0]:
                if image_name != df.iloc[idx - 1, 0]:
                    combined = draw_keypoints(image, keypoints, edges, keypoint_names, boxes=False)
                    combined, PLEAA = cal_bowleg(combined, RtC1, RtC2, LtC1, LtC2)
                    cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
                    f.writerow([image_name, str(int(float(PLEAA[0]))), str(int(float(PLEAA[1])))])
                    reset(RtC1, RtC2, LtC1, LtC2, PLEAA)

                if image_name == df.iloc[idx - 1, 0]:
                    combined = draw_keypoints(combined, keypoints, edges, keypoint_names, boxes=False)
                    combined, PLEAA = cal_bowleg(combined, RtC1, RtC2, LtC1, LtC2)
                    cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
                    f.writerow([image_name, str(int(float(PLEAA[0]))), str(int(float(PLEAA[1])))])                    
                    reset(RtC1, RtC2, LtC1, LtC2, PLEAA)

        if idx == (len(choice_idx)-1):
            if image_name != df.iloc[idx - 1, 0]:
                combined = draw_keypoints(image, keypoints, edges, keypoint_names, boxes=False)
                combined, PLEAA = cal_bowleg(combined, RtC1, RtC2, LtC1, LtC2)
                cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
                f.writerow([image_name, str(int(float(PLEAA[0]))), str(int(float(PLEAA[1])))])
            if image_name == df.iloc[idx - 1, 0]:
                combined = draw_keypoints(combined, keypoints, edges, keypoint_names, boxes=False)
                combined, PLEAA = cal_bowleg(combined, RtC1, RtC2, LtC1, LtC2)
                cv2.imwrite(os.path.join(dst_path, "result_" + image_name), combined)
                f.writerow([image_name, str(int(float(PLEAA[0]))), str(int(float(PLEAA[1])))])
                
## inference
cfg = get_cfg()   
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 0  # On Windows environment, this value must be 0.
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints
cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((num_keypoints, 1), dtype=float).tolist()
cfg.OUTPUT_DIR = output_dir

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_231008.pth")  # 학습된 모델 들어가 있는 곳
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # custom testing threshold
predictor = DefaultPredictor(cfg)

test_list = os.listdir(test_dir)
test_list.sort()
except_list = []

files = []
preds = []
for file in tqdm(test_list):
    filepath = os.path.join(test_dir, file)
    im = cv2.imread(filepath)
    if im is None:
        print(f'Error reading file: {filepath}')
        continue
    outputs = predictor(im)
    for i in range(len(outputs["instances"])):
        pred_keypoints = outputs["instances"][i].to("cpu").get("pred_keypoints").numpy()
        files.append(file)
        pred = []
        try:
            for out in pred_keypoints[0]:
                pred.extend([float(e) for e in out[:2]])
        except IndexError:
            pred.extend([0] * (2 * num_keypoints))
            except_list.append(filepath)
        preds.append(pred)

df_sub = pd.read_csv(column_path)
df = pd.DataFrame(columns=df_sub.columns)
df["image"] = files
df.iloc[:, 1:] = preds

df.to_csv(os.path.join(cfg.OUTPUT_DIR, f"{dataname}_keypoints.csv"), index=False)
if except_list:
    print(
        "The following images are not detected keypoints. The row corresponding that images names would be filled with 0 value."
    )
    print(*except_list)


# save_samples(cfg.OUTPUT_DIR, test_dir, os.path.join(cfg.OUTPUT_DIR, f"{dataname}_keypoints.csv"), mode="choice", size=5, index=range(len(files)))