import pickle
import matplotlib.pyplot as plt
import numpy as np
from circle import circle
from PIL import Image
from archive import Archive

flag = ""
while True:
	flag = input("real, prompt or variation? ").lower().strip()
	if flag == "real" or flag == "prompt" or flag == "variation":
		break
	else:
		print("Error: Answer must be real or dalle2")

if (flag == "real"):
	pa: Archive = Archive.load(Archive.REAL)
elif (flag == "prompt"):
	pa: Archive = Archive.load(Archive.PROMPT)
elif (flag == "variation"):
	pa: Archive = Archive.load(Archive.VARIATION)
circlesToAppend = []

print(pa)
i=0
for C in pa:
	C.estimateCoefficients(image=np.asarray(Image.open(C.image_id), dtype=np.uint8), M=-1)
	i=i+1
	print(i)
print(pa)

pa.save()

