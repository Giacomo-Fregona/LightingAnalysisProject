import pickle
import matplotlib.pyplot as plt
import numpy as np
from circle import circle

def distance(array1, array2):
	return np.sqrt(np.sum((array1 - array2) ** 2))

flag = ""
while True:
	flag = input("real, prompt or variation? ").lower().strip()
	if flag == "real" or flag == "prompt" or flag == "variation":
		break
	else:
		print("Error: Answer must be real or dalle2")

fileToOpen = "./Archive/" + flag + ".pkl"
with open(fileToOpen, "rb") as file:
	data = pickle.load(file)
file.close()

"""
data_dict had the following structure:
	'name_of_the_image': 'name_of_the_coefficient': array with vector of values
"""

# Creating a dict with image_id keys
data_dict = {}
C: circle
for C in data:

	# Initializing data_dict(image_id) if not already existing
	if C.image_id not in data_dict:
		data_dict[C.image_id] = {coeff: [] for coeff in C.coeffList}

	# Adding coefficients of the considered C
	for coeff in circle.coeffList:
		data_dict[C.image_id][coeff].append(C.get_coeff(coeff))

# Printing data_dict
for nome_immagine, coefficient_dict in data_dict.items():
	print(nome_immagine + ":")
	for coefficient_name, C_coeff_RGB in coefficient_dict.items():
		print("\t" + coefficient_name + ":")
		for element in C_coeff_RGB:
			print("\t\t", end="")
			print(element)

# Cycling in the image to fill the standard_deviations dict
standard_deviations = {}
for image_id, values in data_dict.items():

	# Computing standard deviation on every coefficient
	std_dev = {coeff: np.std(values['l00'], axis=0) for coeff in circle.coeffList}

	# Adding the result to the dict
	standard_deviations[image_id] = std_dev

# Printing standard_deviations
for nome_immagine, coefficient_dict in standard_deviations.items():
	print(nome_immagine + ":")
	for coefficient_name, C_coeff_RGB in coefficient_dict.items():
		print("\t" + coefficient_name + ":")
		for element in C_coeff_RGB:
			print("\t\t", end="")
			print(element)


'''
 Test "Figura 4" fatto assieme
'''

#Calcolo coefficienti normalizzati
RGB = {
	'R': {coeff: [] for coeff in circle.coeffList},
	'G': {coeff: [] for coeff in circle.coeffList},
	'B': {coeff: [] for coeff in circle.coeffList}
}

for nome_immagine, coefficient_dict in data_dict.items():

	l00 = coefficient_dict['l00']
	for j in range(len(l00)):

		for coefficient_name, C_coeff_RGB in coefficient_dict.items():

			if coefficient_name != 'l00':
				C_coeff_RGB = [C_coeff_RGB[j][i] / l00[j][i] for i in range(3)]

				RGB['R'][coefficient_name].append(C_coeff_RGB[0])
				RGB['G'][coefficient_name].append(C_coeff_RGB[1])
				RGB['B'][coefficient_name].append(C_coeff_RGB[2])

# Calcolo della mediana
median = {'R': {},
		  'G': {},
		  'B': {}
		  }

for color in ['R', 'G', 'B']:
	for coeff in ['l1m1', 'l10', 'l11', 'l2m2', 'l2m1', 'l20', 'l21', 'l22']:
		l = RGB[color][coeff]
		l.sort()
		if (len(l) % 2) == 0:
			m = (l[int(len(l) / 2)] + l[int(len(l) / 2 + 1)]) / 2
		else:
			m = l[int(len(l) / 2)]

		median[color][coeff] = m

print(median)








# Test Aggiunto da Dario
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
			l1m1List = [[myC.l1m1[k] / myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l10List = [[myC.l10[k] / myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l11List = [[myC.l11[k] / myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l2m2List = [[myC.l2m2[k] / myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l2m1List = [[myC.l2m1[k] / myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l20List = [[myC.l20[k] / myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l21List = [[myC.l21[k] / myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]
			l22List = [[myC.l22[k] / myC.l00[k] for k in range(3)] for myC in listOfCircleInSameImage]

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
