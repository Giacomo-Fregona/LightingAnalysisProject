import pickle
import matplotlib.pyplot as plt
import numpy as np

def distance(array1, array2):
	return np.sqrt(np.sum((array1 - array2) ** 2))

flag = ""
while True:
	flag = input("real, prompt or variation? ").lower().strip()
	if flag == "real" or flag == "prompt" or flag == "variation":
		break
	else:
		print("Error: Answer must be real or dalle2")

fileToOpen = "./Archive/"+flag+".pkl"
with open(fileToOpen, "rb") as file:
	data = pickle.load(file)
file.close()


"""
data_dict had the following structure:
	'name_of_the_image': 'name_of_the_coefficient': array with vector of values
"""
print(data)
print()

listOfCircleInSameImage = [data[0]]
for i in range(1, len(data)):
	if data[i].image_id == listOfCircleInSameImage[0].image_id:
		listOfCircleInSameImage.append(data[i])
	else:
		if len(listOfCircleInSameImage) == 1:
			print(listOfCircleInSameImage[0].image_id)
			print("NON CONTARE: UNA SOLA SFERA NELL'IMMAGINE")
			print()
		else:
			l00List = [myC.l00 for myC in listOfCircleInSameImage]
			l1m1List = [[myC.l1m1[k]/myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l10List = [[myC.l10[k]/myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l11List = [[myC.l11[k]/myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l2m2List = [[myC.l2m2[k]/myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l2m1List = [[myC.l2m1[k]/myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l20List = [[myC.l20[k]/myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l21List = [[myC.l21[k]/myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l22List = [[myC.l22[k]/myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			
			print(listOfCircleInSameImage[0].image_id)
			
			stdl1m1 = np.std(l1m1List, axis=0)
			stdl10 = np.std(l10List, axis=0)
			stdl11 = np.std(l11List, axis=0)
			stdl2m2 = np.std(l2m2List, axis=0)
			stdl2m1 = np.std(l2m1List, axis=0)
			stdl20 = np.std(l20List, axis=0)
			stdl21 = np.std(l21List, axis=0)
			stdl22 = np.std(l22List, axis=0)
			print(f"\tvar l1m1: {stdl1m1}")
			print(f"\tvar l10: {stdl10}")
			print(f"\tvar l11: {stdl11}")
			print(f"\tvar l2m2: {stdl2m2}")
			print(f"\tvar l2m1: {stdl2m1}")
			print(f"\tvar l20: {stdl20}")
			print(f"\tvar l21: {stdl21}")
			print(f"\tvar l22: {stdl22}")
			
			print(f"np.quantile(l1m1List, 0.35, axis=0): {np.quantile(l1m1List, 0.35, axis=0)}")
			print(f"np.quantile(l1m1List, 0.5, axis=0): {np.quantile(l1m1List, 0.5, axis=0)}")
			print(f"np.quantile(l1m1List, 0.65, axis=0): {np.quantile(l1m1List, 0.65, axis=0)}")
			print()
			
		listOfCircleInSameImage = [data[i]]




