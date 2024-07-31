import cv2
import numpy as np
import torch
import time
import pandas as pd
import sys
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A
import matplotlib.pyplot as plt


device = torch.device("cpu")

TRAIN_DATA_PATH = './train/labels.csv'
IMG_PATH = './train/img/'
Columns = ['UpperEyelidX', 'UpperEyelidY', 'RightCornX', 'RightCornY', 'LowerEyelidX', 'LowerEyelidY', 'LeftCornX', 'LeftCornY']

# Define hyperparameters
EPOCHS = 100        # number of epochs
LR = 0.001         # Learning rate
IMG_SIZE = 200     # Size of image

BATCH_SIZE = 32    # Batch size

widthImg = 200
heightImg = 200

df = pd.read_csv(TRAIN_DATA_PATH, sep=';')

train_df, val_df = train_test_split(df, test_size=0.2, random_state=57)

def get_train_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),      # Horizontal Flip with 0.5 probability
        # A.VerticalFlip(p=0.5),         # Vertical Flip with 0.5 probability
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.GaussNoise(var_limit=(10, 100), p=1)
    ],
    keypoint_params=A.KeypointParams(format='xy'),
    is_check_shapes=False)

def exportONNX(model):
    torch_input = torch.randn(1, 1, 200, 200)
    model_path = 'best_model_landmarks.onnx'
    torch.onnx.export(model, torch_input, model_path,
     verbose=True,
     input_names=["input"],
     output_names=["output"],
     opset_version=11)
def get_val_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE)
    ],
    keypoint_params=A.KeypointParams(format='xy'),
    is_check_shapes=False)

class EyeDataset(Dataset):
    def __init__(self, df, augs):
        self.df = df
        self.augs = augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]

        path = sample['Image']
        image = IMG_PATH + path
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # print(row, Columns)
        df_keypoints = sample[Columns]
        image = np.expand_dims(image, axis=-1)

        # Apply augmentations
        if self.augs:
            x_feature_cord = np.array(df_keypoints[0::2].tolist())
            y_feature_cord = np.array(df_keypoints[1::2].tolist())
            keypoints = [(x_feature_cord[i], y_feature_cord[i]) for i in range(len(x_feature_cord))]
            data = self.augs(image=image, keypoints=keypoints)
            if len(data['keypoints']) == len(Columns) // 2:
                image = data['image']
                keypoints = data['keypoints']

                df_keypoints = []
                for x, y in keypoints:
                    df_keypoints.append(x)
                    df_keypoints.append(y)

        # Transpose image dimensions in pytorch format
        # (H,W,C) -> (C,H,W)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        # Normalize the images and masks
        image = torch.Tensor(image) / 255.0
        keypoints = torch.Tensor(df_keypoints)

        return image, keypoints

class LandmarkModel(nn.Module):
    def __init__(self):
        super(LandmarkModel, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, (3, 3), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, (3, 3), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, (3, 3), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2)),

            nn.Flatten(),
            nn.Linear(25600, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 8),
        )

    def forward(self, images, keypoints=None):
        logits = self.model(images)

        if keypoints != None:
            loss = nn.MSELoss()(logits, keypoints)
            return logits, loss
        return logits

val_data = EyeDataset(val_df, get_val_augs())
model = LandmarkModel()
model.load_state_dict(torch.load('best_model_landmarks.pt', map_location=torch.device('cpu')))

onnx_model_path = 'best_model_landmarks.onnx'


def make_inference(idx):
    image, keypoints = val_data[idx]

    image = image.to(device).unsqueeze(0)
    keypoints = keypoints.to(device).unsqueeze(0)
    t1 = time.time()
    pred_keypoints, loss = model(image, keypoints=keypoints) # (C, H, W) -> (1, C, H, W)
    t2 = time.time()
    t = t2 - t1
    return image, pred_keypoints, keypoints, loss, t

mi_t = np.empty(0)
for i in np.random.randint(0, len(val_data), 20):
    image, pred_keypoints, keypoints, loss, t = make_inference(i)
    mi_t = np.append(mi_t, t)
#     print('orig: ', keypoints)
#     print("predict: ", pred_keypoints)

CV = np.empty(0)
for i in np.random.randint(0, len(val_data), 20):
    opencv_net = cv2.dnn.readNetFromONNX(onnx_model_path)

    image, keypoints = val_data[i]
    input_img = image[0].numpy()
    input_img = cv2.resize(input_img, (200, 200))

    input_blob = cv2.dnn.blobFromImage(image=input_img, size=(200, 200))
    opencv_net.setInput(input_blob)

    t1 = time.time()
    out = opencv_net.forward()
    t2 = time.time()

    CV = np.append(CV, t2-t1)

    # print(f"CV inference Duration work = {t2 - t1}")
    # print('orig: ', keypoints)
    # print("predict: ", out)

import onnxruntime

session = onnxruntime.InferenceSession('best_model_landmarks.onnx', None, providers=['CPUExecutionProvider'])

ORT = np.empty(0)
for i in np.random.randint(0, len(val_data), 20):

    image, keypoints = val_data[i]
    input_img = image[0].numpy()
    input_img = [[cv2.resize(input_img, (200, 200))]]

    input_nodes = session.get_inputs()
    input_names = [node.name for node in input_nodes]
    input_shapes = [node.shape for node in input_nodes]
    input_types = [node.type for node in input_nodes]
    output_nodes = session.get_outputs()
    output_names = [node.name for node in output_nodes]
    output_shapes = [node.shape for node in output_nodes]
    output_types = [node.type for node in output_nodes]

    t1 = time.time()

    output_tensors = session.run([],
                                  input_feed={input_names[0]: input_img},
                                  run_options=None)
    t2 = time.time()

    ORT = np.append(ORT, t2 - t1)

    # print(f"ORT inference Duration work = {t2 - t1}")
    # print('orig: ', keypoints)
    # print("predict: ", output_tensors)

from openvino.runtime import Core

ie = Core()

model_onnx = ie.read_model(model=onnx_model_path)
compiled_model = ie.compile_model(model=model_onnx, device_name="CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

OV = np.empty(0)
for i in np.random.randint(0, len(val_data), 20):
    image, keypoints = val_data[i]
    input_data = np.expand_dims(image[0], axis=(0, 1)).astype(np.float32)

    t1 = time.time()
    request = compiled_model.create_infer_request()
    request.infer(inputs={input_layer.any_name: input_data})
    result = request.get_output_tensor(output_layer.index).data
    t2 = time.time()
    OV = np.append(OV, t2 - t1)
    print(i, result, keypoints)

# print('make_inference inference: ', np.mean(mi_t))
# print('OpenCV Python API inference: ', np.mean(CV))
# print('OnnxRuntime inference: ', np.mean(ORT))
# print('OpenVINO inference: ', np.mean(OV))