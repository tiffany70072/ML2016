from PIL import Image
import sys

image = Image.open(sys.argv[1]) 
image = image.rotate(180)
image.save('ans2.png')