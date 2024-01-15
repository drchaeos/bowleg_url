import os
import glob
import csv
import traceback
import pandas as pd
import numpy as np
from tqdm import tqdm
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import cv2
import json
import shutil
from bowleg_inference import save_samples, get_cfg, DefaultPredictor, model_zoo 
from correction import correction
from view_xray import view_xray

app = FastAPI()


MODEL_DIR = "./out"
column_DIR = './columns/bow_leg.csv'


# Mounting directories
app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/columns", StaticFiles(directory="columns"), name="columns")
app.mount("/xraysample", StaticFiles(directory="xraysample"), name="xraysample")


# Configuration for keypoint prediction


def analyze_image(image_path, save_path, model_path, column_path, filename):
    num_keypoints = 4
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = num_keypoints
    cfg.TEST.KEYPOINT_OKS_SIGMAS = np.ones((num_keypoints, 1), dtype=float).tolist()
    cfg.OUTPUT_DIR = model_path
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_231008.pth")  # 학습된 모델 들어가 있는 곳
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # custom testing threshold
    predictor = DefaultPredictor(cfg)
    test_list = os.listdir(image_path)
    test_list = [filename]
    except_list = []

    files = []
    preds = []
    for file in tqdm(test_list):
        filepath = os.path.join(image_path, file)
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

    df.to_csv(os.path.join(save_path, "bow_leg_keypoints.csv"), index=False)
    if except_list:
        print(
            "The following images are not detected keypoints. The row corresponding that images names would be filled with 0 value."
        )
        print(*except_list)

        files = []

    save_samples(save_path, image_path, os.path.join(save_path, "bow_leg_keypoints.csv"), mode="choice", size=5, index=range(len(files)))



@app.post("/upload/")
@app.post("/upload")
async def analysis_lateral(request: Request, file: UploadFile = File(...)):
    base_url = "http://localhost:8000" 

    contents = await file.read()
    filename = file.filename
    output_filename = f"result_{filename}"

    foldname = os.path.splitext(filename)[0]
    UPLOAD_DIR = f"images/{foldname}"
    OUTPUT_DIR = f"results/{foldname}"
    Xray_DIR = 'xraysample'

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    with (Path(UPLOAD_DIR) / filename).open("wb") as buffer:
        buffer.write(contents)
   
    try:
        analyze_image(UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR, column_DIR, filename)
        Xray_image, Rt_angle, Lt_angle, Rt_shape, Lt_shape = view_xray(Xray_DIR, os.path.join(OUTPUT_DIR, 'PLEAA_result.csv'))
        cor_image = correction(os.path.join(OUTPUT_DIR, 'bow_leg_keypoints.csv'), os.path.join(OUTPUT_DIR, 'PLEAA_result.csv'),
                                os.path.join(UPLOAD_DIR, filename), OUTPUT_DIR)
        

        return {
            "imageUrl": f"{base_url}/{UPLOAD_DIR}/{filename}",
            "analysis": {
            "image": f"{base_url}/{OUTPUT_DIR}/{output_filename}",
            "xray": f"{base_url}/{Xray_image}",
            "left-type": Lt_shape,
            "left-angle": Lt_angle,
            "right-type": Rt_shape,
            "right-angle": Rt_angle,
        }, 
        "correction": {
            "image": f"{base_url}/{cor_image}",
            "left-angle": Lt_angle,
            "right-angle": Rt_angle,
        }
}
    except Exception as e:
        trace = traceback.format_exc()
        print(trace)
        return {
            "image_result_src": f"{base_url}/{OUTPUT_DIR}/{output_filename}"
        }



@app.get("/clear/")
@app.get("/clear")
async def clear():
    folders_to_clear = ["images", "results"]
    
    for folder in folders_to_clear:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                # 파일이면 삭제
                if os.path.isfile(path):
                    os.remove(path)
                # 폴더면 폴더 전체를 삭제
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                return {"status": "error", "message": f"Failed to delete {path}. Reason: {str(e)}"}

    return {"status": "success", "message": "All images and results have been cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
