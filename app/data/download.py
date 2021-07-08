import wget
import zipfile
import os
import shutil
import argparse



def get_path_name():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    args = parser.parse_args()

    return str(args.path)


def clean_up(path):
    path_dir = os.path.join(path)
    if os.path.exists(path_dir):
        shutil.rmtree(path_dir)
        os.makedirs(path_dir)
    return path_dir


def download(path):
    clean_up(path)
    wget.download(
        "https://drive.google.com/uc?id=1WcKcGDThi0j3q9v_-pW3q0PuQL1wTSOK", path+"/fra-eng.zip")
    file = zipfile.ZipFile(path+"/fra-eng.zip")
    file.extractall(path)


if __name__ == "__main__":

    path = get_path_name()

    download(path)
