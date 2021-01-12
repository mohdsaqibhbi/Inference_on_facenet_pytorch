from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
import torch
import os
import pickle
import cv2
import matplotlib.pyplot as plt
from PIL import Image

workers = 0 if os.name == 'nt' else 4

import warnings
warnings.filterwarnings("ignore")

# Load Models
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
image_size=160, margin=0, min_face_size=20,
thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Get Embeddings
def get_embeddings(image):

    name = image.split('/')[-1].split('.')[0]
    try:
        face = mtcnn(Image.open(image))
    except:
        print("Couldn't read the image ", name)

    try:
        return resnet(face.unsqueeze(0).to(device)).detach(), name
    except Exception:
        print("Couldn't get the embeddings of image ", name)
        return None, None