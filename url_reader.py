from PIL import Image
import numpy as np
import io,urllib
#if you want to print the entire numpy array uncomment bellow
# np.set_printoptions(threshold=np.nan)

url_name = raw_input('Image url?\n')
file = io.BytesIO(urllib.urlopen(url_name).read())
img = Image.open(file)

#show our initial image gotten from url
img.show()

#make iamge into array
img_ar = np.array(img.getdata(),np.uint8).reshape(img.size[1], img.size[0], 3)
print(img_ar)

#testing make it redder
for i in img_ar:
    for j in i:
        x = j[0] + 50
        if x < 255:
            j[0] = x
        else:
            j[0] = 255

#make array back to image to display and show
img2 = Image.fromarray(img_ar)
img2.show()

#attempt to crop
#don't have to worry about the image being smaller than the amount you crop python will just do till its limit then.
img3 = Image.fromarray(img_ar[:1000][:1000])
img3.show()
