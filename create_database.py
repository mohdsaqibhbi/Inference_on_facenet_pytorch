from utils import get_embeddings
import os
import torch
import pickle
import argparse

def create_database(in_path, out_path):

    images_list = os.listdir(in_path)
    embeddings_set = torch.rand(len(images_list), 1, 512)
    id_to_name = {}
    for i, image in enumerate(images_list):
        embeddings, name = get_embeddings(os.path.join(in_path, image))
        if embeddings is not None:
            embeddings_set[i] = embeddings
            id_to_name[i] = name
    database = [embeddings_set, id_to_name]

    with open(out_path,"wb") as pkl_out:
        pickle.dump(database, pkl_out)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-in", "--in_path",
                    required=True,
                    help="path of the input images")
    ap.add_argument("-o", "--out_path",
                    default = "database.pkl",
                    help="path of database to be saved")

    args = vars(ap.parse_args())

    create_database(args["in_path"], args["out_path"])