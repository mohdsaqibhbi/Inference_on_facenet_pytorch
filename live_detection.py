from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1
import torch
import pickle
import cv2
import os
import argparse

workers = 0 if os.name == 'nt' else 4

import warnings
warnings.filterwarnings("ignore")

def live_detection(database, threshold):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
    )

    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    with open(database, "rb") as pkl_in:
        database = pickle.load(pkl_in)

    embeddings_set, id_to_name = database

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            face = mtcnn(rgb)
            c, _ = mtcnn.detect(rgb)
            c = c.flatten().tolist()
            embeddings = resnet(face.unsqueeze(0).to(device)).detach()
            if embeddings is not None:
                index = (embeddings_set - embeddings).norm(dim=-1).argmin().item()
                dist = (embeddings_set - embeddings).norm(dim=-1).min().item()
                if dist < threshold:
                    name = id_to_name[index]
                else:
                    name = "Unknown"

            cv2.rectangle(frame,(int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (0,0,255), 2)
            cv2.putText(frame, name, (int(c[0])-120,int(c[1])-10), cv2.FONT_HERSHEY_TRIPLEX, 1.0, [0,0,255], 2, 1)
        except Exception:
            pass


        cv2.imshow('Input', frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--database",
                    required=True,
                    help="path to the database")
    ap.add_argument("-th", "--threshold",
                    type=float,
                    default=1.0,
                    help="theshold to euclidean distance")

    args = vars(ap.parse_args())

    live_detection(args["database"], args["threshold"])