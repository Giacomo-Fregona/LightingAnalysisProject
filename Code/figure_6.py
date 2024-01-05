from matplotlib import pyplot as plt

from circle import circle
from archive import Archive

def get_couples(l: list):
	"""
	getting all the couples of circles to be compared
	@param l: list of the circles belonging to the same image
	@return: list of tuples representing couples of circles
	"""
	output = []
	for i in range(len(l)-1):
		for j in range(i+1, len(l)):
			output.append((l[i],l[j]))
	return output

def ax_index(coeff: str):
	"""
	Getting the right ax for a given coefficient
	@param coeff: the coefficient
	@return: the index of the axis array
	"""

	# Checking if coeff is a valid input
	if not (coeff in circle.coeff_list):
		raise Exception(f'coeff {coeff} not in coeff list')

	# returning the right axes index
	if coeff == circle.coeff_list[0]:
		return 0
	elif coeff in circle.coeff_list[1:4]:
		return 1
	elif coeff in circle.coeff_list[4:]:
		return 2
	else:
		raise Exception(f'Inappropriate behaviour of ax_index function')


m = []
M = []

def diagonal_line(coeff: str, xm, xM, axes):
	"""
	Handling the diagonal line of axes
	@param coeff: the coefficient we are working with
	@param xm: the minimum value of the data to be added
	@param xM: the maximum value of the data to be added
	@param axes: the actual axes array
	@return: None
	"""

	# Checking if coeff is a valid input
	if not (coeff in circle.coeff_list):
		raise Exception(f'coeff {coeff} not in coeff list')

	# Adding min and max to the lists
	global m, M
	m.append(xm)
	M.append(xM)

	# In the case coefficients are 0, 3 or 8, plot the diagonal line and reset the m, M lists
	if coeff == circle.coeff_list[0] or coeff == circle.coeff_list[3] or coeff == circle.coeff_list[8]:
		m = min(m)
		M = max(M)
		axes[ax_index(coeff)].plot([m, M], [m, M], color='black', linestyle='--')
		m = []
		M = []

def coeff_pedix(coeff: str):
	"""
	From coeff to its pedix (for label representation)
	@param coeff: the coeff
	@return: the pedix string as output
	"""

	# Case of minus ('m' replaced with '-')
	if len(coeff) == 4:
		minus = '-'
	else:
		minus = ''

	return '{' + coeff[1] + ',' + minus + coeff[-1] + '}'

if __name__ == '__main__':

	for A_filename in [Archive.REAL, Archive.PROMPT]:

		# Loading the archive
		A: Archive = Archive.load(A_filename)
		A_dict = A.as_dict()

		fig, axes = plt.subplots(1, 3, figsize=(20, 5))

		# Retrieving all possible couples inside the same image
		couples = [couple for circles_list in A_dict.values() for couple in get_couples(circles_list)]
		print(f'{len(couples)} couples')

		# Getting the data
		data_1 = {coeff: [] for coeff in circle.coeff_list}
		data_2 = {coeff: [] for coeff in circle.coeff_list}
		for (C1, C2) in couples:
			for coeff in circle.coeff_list:

				data_1[coeff].append(C1.get_coeff(coeff))
				data_2[coeff].append(C2.get_coeff(coeff))

		# Plotting
		for coeff in circle.coeff_list:

			# Retrieving data (only red)
			# x = [RGB[0] for RGB in data_1[coeff]]
			# y = [RGB[0] for RGB in data_2[coeff]]

			# Normalized version
			x = [RGB[0] / (data_1['l00'][i][0]) for i, RGB in enumerate(data_1[coeff])]
			y = [RGB[0] / (data_2['l00'][i][0]) for i, RGB in enumerate(data_2[coeff])]


			# Plotting the scatter
			axes[ax_index(coeff)].scatter(x, y, label=rf'$Y_{coeff_pedix(coeff)}$')

			# Handling diagonal line
			diagonal_line(coeff, min(x), max(x), axes)

		# Add labels and titles
		fig.suptitle(f'{A_filename}')
		axes[0].set_title('Zeroth order harmonics')
		axes[1].set_title('First order harmonics')
		axes[2].set_title('Second order harmonics')
		for i in range(3):
			axes[i].set_xlabel('Lighting coefficients')
			axes[i].set_ylabel('Lighting coefficients')
			axes[i].legend()

	plt.show()
