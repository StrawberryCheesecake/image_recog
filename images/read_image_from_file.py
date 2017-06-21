import urllib

file = open("images.txt","r+")
i = 0
for line in file:
    try:
        urllib.urlretrieve(line, "seatbelt"+str(i)+".jpg")
        i += 1
    except:
        continue
file.close()
