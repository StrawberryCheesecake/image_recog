import download_images as dli
import create_tfrecord as ctf
import os
import sys
import  glob

def main():
    if len(sys.argv) == 3:
        class_name = sys.argv[1]
        url_file = sys.argv[2]
    else:
        sys.exit("Not enough arguments provided.")
    dli.run(class_name,url_file)
    os.remove("./images/labels.txt")
    for filename in glob.glob("./images/0x17*"):
        os.remove(filename)

    ctf.run(tfrecord_filename="0x17",dataset_dir="./images")


if __name__ == '__main__':
    main()
