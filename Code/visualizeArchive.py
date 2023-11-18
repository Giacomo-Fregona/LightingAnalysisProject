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
data_dict has the following structure:
	'name_of_the_image': 'name_of_the_coefficient': array with vector of values
"""
data_dict = {}

for item in data:
	for key, values in item.items(): #key = nome immagine, values = C
		if key not in data_dict:
			data_dict[key] = {
				'l00': [],
				'l1m1': [],
				'l10': [],
				'l11': [],
				'l2m2': [],
				'l2m1': [],
				'l20': [],
				'l21': [],
				'l22': []
			}
		data_dict[key]['l00'].append(item[key].l00)
		data_dict[key]['l1m1'].append(item[key].l1m1)
		data_dict[key]['l10'].append(item[key].l10)
		data_dict[key]['l11'].append(item[key].l11)
		data_dict[key]['l2m2'].append(item[key].l2m2)
		data_dict[key]['l2m1'].append(item[key].l2m1)
		data_dict[key]['l20'].append(item[key].l20)
		data_dict[key]['l21'].append(item[key].l21)
		data_dict[key]['l22'].append(item[key].l22)

for nome_immagine, coefficient_dict in data_dict.items():
	print(nome_immagine + ":")
	for coefficient_name, C_coeff_RGB in coefficient_dict.items():
		print("\t" + coefficient_name + ":")
		for element in C_coeff_RGB:
			print("\t\t", end="")
			print(element)


standard_deviations = {}

for key, values in data_dict.items():
	std_dev = {
		'l00': np.std(values['l00'], axis=0),
		'l1m1': np.std(values['l1m1'], axis=0),
		'l10': np.std(values['l10'], axis=0),
		'l11': np.std(values['l11'], axis=0),
		'l2m2': np.std(values['l2m2'], axis=0),
		'l2m1': np.std(values['l2m1'], axis=0),
		'l20': np.std(values['l20'], axis=0),
		'l21': np.std(values['l21'], axis=0),
		'l22': np.std(values['l22'], axis=0)
	}
	
	standard_deviations[key]= std_dev

for nome_immagine, coefficient_dict in standard_deviations.items():
	print(nome_immagine + ":")
	for coefficient_name, C_coeff_RGB in coefficient_dict.items():
		print("\t" + coefficient_name + ":")
		for element in C_coeff_RGB:
			print("\t\t", end="")
			print(element)



# Test "Figura 4"

#Calcolo coefficienti normalizzati
RGB = {'R':{
		'l1m1': [],
		'l10': [],
		'l11': [],
		'l2m2': [],
		'l2m1': [],
		'l20': [],
		'l21': [],
		'l22': []
	},
	'G': {
		'l1m1': [],
		'l10': [],
		'l11': [],
		'l2m2': [],
		'l2m1': [],
		'l20': [],
		'l21': [],
		'l22': []
	},
	'B': {
		'l1m1': [],
		'l10': [],
		'l11': [],
		'l2m2': [],
		'l2m1': [],
		'l20': [],
		'l21': [],
		'l22': []
	}
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
median = {'R':{},
	'G': {},
	'B': {}
}

for color in ['R', 'G', 'B']:
	for coeff in ['l1m1', 'l10', 'l11', 'l2m2', 'l2m1', 'l20', 'l21', 'l22']:
		l = RGB[color][coeff]
		l.sort()
		if (len(l)%2) == 0:
			m = (l[int(len(l)/2)] + l[int(len(l)/2+1)])/2
		else:
			m = l[int(len(l)/2)]

		median[color][coeff] = m

print(median)


