# Inference_on_facenet_pytorch
This repo is about doing inferencing on pretrained model taken from [facenet-pytorch](https://github.com/timesler/facenet-pytorch).
Running the pretrained model on an image return am embeddings of the person's face with size (1, 512). These embeddings can be used as per requirement.

## How to run
- Clone the repo using `git clone https://github.com/mohdsaqibhbi/Inference_on_facenet_pytorch.git`.
- Go to this directory using `cd Inference_on_facenet_pytorch`.
- Create virtual environment.
- Install dependencies using `pip install -r requirements.txt`.
- Create the database and then use live face detection to test it.
- To create the database of embeddings
    - Put the images in the folder `data/database/images/`
    - Run the command `python create_database.py -in data/database/images/ -o data/database/database.pkl`
- To update the database with new person's embeddings
    - Run the command `python update_database.py -in data/database/database.pkl -i data/Chadwick_Boseman.jpg -n Chadwick_Boseman.jpg"`
- For live face detection
    - Need to create database embeddings first.
    - Run the command `python live_detection.py -d data/database/database.pkl -th 1.0`
- To understand step by step, how to create database, update database, face verification, face recognition and live face detection, follow the jupyter-notebook [Face_Recognition](Face_Recognition.ipynb).

## Command line arguments
- **create_database.py**
    | tag (* = required)| variable          | options                                        | default value   |
    |:-----------------:|:------------------|:-----------------------------------------------|:----------------|
    | -in *             | in_path           | path of the input images                       | REQUIRED        |
    | -o                | out_path          | path of database to be saved                   | "database.pkl"  |
    
- **update_database.py**
    | tag (* = required)| variable          | options                                        | default value   |
    |:-----------------:|:------------------|:-----------------------------------------------|:----------------|
    | -in *             | in_path           | path of the input database                     | REQUIRED        |
    | -i *              | image             | path of input image                            | REQUIRED        |
    | -n                | name              | name of the person                             | None            |
    
    **Note** : If name is not given, image name will be used as person's name.
    
- **live_detection.py**
    | tag (* = required)| variable          | options                                        | default value   |
    |:-----------------:|:------------------|:-----------------------------------------------|:----------------|
    | -d *              | database          | path of the database                           | REQUIRED        |
    | -th               | threshold         | threshold to euclidean distance                | 1.0             |
