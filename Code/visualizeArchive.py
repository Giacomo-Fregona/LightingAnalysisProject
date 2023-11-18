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





