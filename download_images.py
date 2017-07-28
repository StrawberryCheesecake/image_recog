import urllib2
import urllib
import urlparse
import os
import socket
import sys
import numpy
import random
from PIL import Image


def download_images(url_file, class_name):

    # Create directory where we save the images for this class
    class_path = "./images/dataset_photos/" + class_name

    if not os.path.exists(class_path):
        os.makedirs(class_path)

    socket.setdefaulttimeout(5)

    filepaths = []

    # Download the images
    with open("./image_urls.txt", "r") as f:
        for x in f:
            urlParts = urlparse.urlsplit(x.strip())
            filename = os.path.join(class_path, urlParts.path.split('/')[-1])

            try:
                urllib.urlretrieve(x.strip(), filename)
            except:
                pass

            print(filename)
            filepaths.append(filename)

    print("Finished downloading images from urls.")
    return filepaths


def validate_images(filepaths):

    validated_filepaths = []

    for filename in filepaths:

        print(filename)

        if os.path.exists(filename):

            # Remove all non jpg files
            if not filename.endswith('.jpg'):
                os.remove(filename)
                continue

            # Remove corrupt jpg files
            try:
                img = Image.open(filename)
                img.verify()
                validated_filepaths.append(filename)

            except (IOError, SyntaxError) as e:
                print('Bad file', filename)
                os.remove(filename)


    print("Finished validating all image files.")
    return validated_filepaths


def remove_empties(filepaths):

    for filename in filepaths:

        if os.path.exists(filename):

            print(filename)

            try:
                # Try open it- if it can't be open, its corrupt
                img = Image.open(filename)
                pix = img.load()

                # If the top left corner is pure white, its an empty
                if pix[0,0] == 238:
                    os.remove(filename)

            except(IOError) as e:
                print("Bad file: ", filename)
                os.remove(filename)


    print("Finished removing empties.")
    return

def run(class_name, url_file):

    # # Script argument validation
    # if len(sys.argv) == 3:
    #     class_name = sys.argv[1]
    #     url_file = sys.argv[2]
    # else:
    #     sys.exit("Not enough arguments provided.")

    if not os.path.exists(url_file):
        sys.exit("Invalid filepath : ", url_file)


    # Download all provided urls and get their filepaths
    paths_to_validate = download_images(url_file, class_name)

    # Remove corrupt, nonexistent, or empty filepaths
    validated_paths = validate_images(paths_to_validate)

    # Remove empty jpegs
    remove_empties(validated_paths)
