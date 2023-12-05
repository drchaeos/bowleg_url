from rembg import remove
from PIL import Image
import pandas as pd
import numpy as np
import cv2


def correction(keypoint_csv_path, PLEAA_csv_path, image_path, save_path):

    # parameter
    keypoint_csvfile = keypoint_csv_path       # bow_leg_keypoints.csv 파일 있는 경로
    PLEAA_csvfile = PLEAA_csv_path               # 결과 각도값 적힌 PLEAA_result.csv 파일 있는 경로
    image_name = pd.read_csv(PLEAA_csvfile).iloc[0, 0]
    target_image = image_path           # 삽입 이미지(교정, 원본)
    edit_image = save_path + f'/edit_{image_name}'               # 교정 후 이미지 경로

    # load keypoint, angle
    keypoint = pd.read_csv(keypoint_csvfile)
    if int(keypoint.iloc[0,3]) < int(keypoint.iloc[1,3]):                               
        Rt_MK = (int(keypoint.iloc[0,3]), int(keypoint.iloc[0,4]))
        Lt_MK = (int(keypoint.iloc[1,3]), int(keypoint.iloc[1,4]))
    if int(keypoint.iloc[0,3]) > int(keypoint.iloc[1,3]):
        Rt_MK = (int(keypoint.iloc[1,3]), int(keypoint.iloc[1,4]))
        Lt_MK = (int(keypoint.iloc[0,3]), int(keypoint.iloc[0,4]))

    angle = pd.read_csv(PLEAA_csvfile)
    Rt_PLEAA = float(angle.iloc[0, 1])
    Lt_PLEAA = float(angle.iloc[0, 2])

    # Load the original image
    img = Image.open(target_image)
    rem = remove(img)
    rem = np.array(rem)
    img = cv2.cvtColor(rem, cv2.COLOR_RGB2BGR)        
    h, w, c = img.shape


    rows, cols = img.shape[:2]
    Rt_M = cv2.getRotationMatrix2D((cols/2, rows/2), -Rt_PLEAA, 1)
    Rt_corrected_image = cv2.warpAffine(img, Rt_M, (cols, rows))

    Lt_M = cv2.getRotationMatrix2D((cols/2, rows/2), Lt_PLEAA, 1)
    Lt_corrected_image = cv2.warpAffine(img, Lt_M, (cols, rows))


    # Define the coordinates of the area to be cropped (x, y, width, height)
    Rt_x, Rt_y, Rt_width, Rt_height = 0, Rt_MK[1], int((Rt_MK[0] + Lt_MK[0]) / 2), h - Rt_MK[1]
    Lt_x, Lt_y, Lt_width, Lt_height = int((Rt_MK[0] + Lt_MK[0]) / 2), Lt_MK[1], w - int((Rt_MK[0] + Lt_MK[0]) / 2), h - Rt_MK[1]

    # Crop the specific area from the original image
    if Rt_PLEAA >= 0:
        Rt_cropped_area = Rt_corrected_image[Rt_y+(int(abs(Rt_PLEAA))+1):h, Rt_x+(int(Rt_PLEAA)+1):Rt_x+Rt_width]
        img[Rt_y:h-(int(abs(Rt_PLEAA))+1), Rt_x:Rt_x-(int(Rt_PLEAA)+1)+Rt_width] = Rt_cropped_area
    if Rt_PLEAA < 0:
        Rt_cropped_area = Rt_corrected_image[Rt_y+(int(abs(Rt_PLEAA))+1):h, Rt_x:Rt_x+Rt_width]
        img[Rt_y:h-(int(abs(Rt_PLEAA))+1), Rt_x+((int(abs(Rt_PLEAA))+1) * 2):Rt_x+((int(abs(Rt_PLEAA))+1) * 2)+Rt_width] = Rt_cropped_area
    if Lt_PLEAA >= 0:
        Lt_cropped_area = Lt_corrected_image[Lt_y+(int(abs(Lt_PLEAA))+1):h, Lt_x:w-(int(Lt_PLEAA)+1)]
        img[Lt_y:h-(int(abs(Lt_PLEAA))+1), Lt_x+(int(Lt_PLEAA)+1):w] = Lt_cropped_area
    if Lt_PLEAA < 0: 
        Lt_cropped_area = Lt_corrected_image[Lt_y+(int(abs(Lt_PLEAA))+1):h, Lt_x:w]
        img[Lt_y:h-(int(abs(Lt_PLEAA))+1), Lt_x-((int(abs(Lt_PLEAA))+1) * 2):Lt_x-((int(abs(Lt_PLEAA))+1) * 2)+Lt_width] = Lt_cropped_area

    cv2.imwrite(edit_image, img)
    cor_image = edit_image

    return cor_image
