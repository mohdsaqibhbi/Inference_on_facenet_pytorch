from utils import get_embeddings
import pickle
import torch
import argparse

def update_database(in_path, image, name):

    with open(in_path, "rb") as pkl_in:
        database = pickle.load(pkl_in)

    embeddings_set, id_to_name = database

    if name:
        embeddings, _ = get_embeddings(image)
    else:
        embeddings, name = get_embeddings(image)

    if embeddings is not None:
        embeddings_set = torch.cat((embeddings_set, embeddings.reshape(1, 1, -1)), dim=0)
        id_to_name[len(id_to_name)] = name

    database = [embeddings_set, id_to_name]

    with open(in_path,"wb") as pkl_out:
        pickle.dump(database, pkl_out)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-in", "--in_path",
                    required=True,
                    help="path of the input database")
    ap.add_argument("-i", "--image",
                    required=True,
                    help="path of input image")
    ap.add_argument("-n", "--name",
                    default=None,
                    help="name of the person")

    args = vars(ap.parse_args())

    update_database(args["in_path"], args["image"], args["name"])