from PIL import Image
import numpy as np
import io,urllib
import json
import signal
import sys
#if you want to print the entire numpy array uncomment bellow
# np.set_printoptions(threshold=np.nan)

#signal handler to exit script
def signal_handler(signal, frame):
    print('Exiting and saving')
    json_file = open('initial_photo_store.txt','w+')
    to_write = json.dumps(data)
    json_file.write(to_write)
    json_file.close()
    sys.exit(0)

if __name__ == '__main__':
    #regist the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    #load already written images to our list
    json_file = open('initial_photo_store.txt','r')
    json_data = json_file.read()
    json_file.close()

    try:
        #if it previously existed and we have json it loads it
        data = json.loads(json_data)
    except:
        #if it doesnt it throws error so we don't bother loading
        data = []

    #while loop to add images to our list
    while(True):
        image_name = raw_input('Image Name:\n')
        url_name = raw_input('Image url:\n')
        try:
            file = io.BytesIO(urllib.urlopen(url_name).read())
            img = Image.open(file)
            img.close()
        except:
            print('incorrect image url please try again')
            continue
        temp_seatbelt = raw_input('Is the person wearing a seatbelt?(y/n only)\n')
        if ( temp_seatbelt != 'y' and temp_seatbelt != 'n'):
            print("incorrect input please use only y/n")
            continue
        elif (temp_seatbelt == 'y'):
            isseatbelt = True
        else:
            isseatbelt = False
        img_dict = {'name':image_name,'url':url_name,'isseatbelt':isseatbelt}
        data.append(img_dict)
