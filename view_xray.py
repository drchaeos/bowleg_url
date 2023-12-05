import pandas as pd
import csv

def view_xray(xray_path, PLEAA_csv_path):
    angle = pd.read_csv(PLEAA_csv_path)
    Rt_PLEAA = int(angle.iloc[0,1])
    Lt_PLEAA = int(angle.iloc[0,2])

    if (-3) <= Rt_PLEAA <= 3:
        Rt_leg = "N"
    if Rt_PLEAA < (-3):
        Rt_leg = "X"
    if Rt_PLEAA > 3:
        Rt_leg = "O"
    if (-3) <= Lt_PLEAA <= 3:
        Lt_leg = "N"
    if Lt_PLEAA < (-3):
        Lt_leg = "X"
    if Lt_PLEAA > 3:
        Lt_leg = "O"
    
    leg = Rt_leg + Lt_leg    
    xray_image = xray_path + f"/{leg}.jpg"

    if Rt_PLEAA >= 0:
        Rt_leg = "O"
    if Rt_PLEAA < 0:
        Rt_leg = "X"
    if Lt_PLEAA >= 0:
        Lt_leg = "O"
    if Lt_PLEAA < 0:
        Lt_leg = "X"

    return xray_image, Rt_PLEAA, Lt_PLEAA, Rt_leg, Lt_leg
