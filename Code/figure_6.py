from matplotlib import pyplot as plt

from circle import circle
from archive import Archive


def getCouples(l: list):
	"""
	getting all the couples of circles to be compared
	@param l: list of the circles belonging to the same image
	@return: list of tuples representing couples of circles
	"""
	output = []
	for i in range(len(l) - 1):
		for j in range(i + 1, len(l)):
			output.append((l[i], l[j]))
	return output


def axIndex(coeff: str):
	"""
	Getting the right ax for a given coefficient
	@param coeff: the coefficient
	@return: the index of the axis array
	"""

	# Checking if coeff is a valid input
	if not (coeff in circle.coeffList):
		raise Exception(f'coeff {coeff} not in coeff list')

	# returning the right axes index
	if coeff == circle.coeffList[0]:
		return 0
	elif coeff in circle.coeffList[1:4]:
		return 1
	elif coeff in circle.coeffList[4:]:
		return 2
	else:
		raise Exception(f'Inappropriate behaviour of ax_index function')


# diagonalLine working variables
m = []
M = []


def diagonalLine(coeff: str, xm, xM, axes):
	"""
	Handling the diagonal line of axes
	@param coeff: the coefficient we are working with
	@param xm: the minimum value of the data to be added
	@param xM: the maximum value of the data to be added
	@param axes: the actual axes array
	@return: None
	"""

	# Checking if coeff is a valid input
	if not (coeff in circle.coeffList):
		raise Exception(f'coeff {coeff} not in coeff list')

	# Adding min and max to the lists
	global m, M
	m.append(xm)
	M.append(xM)

	# In the case coefficients are 0, 3 or 8, plot the diagonal line and reset the m, M lists
	if coeff == circle.coeffList[0] or coeff == circle.coeffList[3] or coeff == circle.coeffList[8]:
		m = min(m)
		M = max(M)
		axes[axIndex(coeff)].plot([-150, 150], [-150, 150], color='black', linestyle='--')
		# axes[axIndex(coeff)].plot([m, M], [m, M], color='black', linestyle='--')
		m = []
		M = []


def coeffPedix(coeff: str):
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

	for filename in [Archive.REAL, Archive.PROMPT, Archive.VARIATION]:

		# Loading the archive
		A: Archive = Archive.load(filename)
		A_dict = A.asDict()

		fig, axes = plt.subplots(1, 3, figsize=(20, 5))

		# Retrieving all possible couples inside the same image
		couples = [couple for circlesList in A_dict.values() for couple in getCouples(circlesList)]
		print(f'{len(couples)} couples')

		# Getting the data
		data1 = {coeff: [] for coeff in circle.coeffList}
		data2 = {coeff: [] for coeff in circle.coeffList}
		for (C1, C2) in couples:
			for coeff in circle.coeffList:
				data1[coeff].append(C1.get_coeff(coeff))
				data2[coeff].append(C2.get_coeff(coeff))

		# Plotting
		for coeff in circle.coeffList:
			# Retrieving data (only red)
			# x = [RGB[1] for RGB in data1[coeff]]
			# y = [RGB[1] for RGB in data2[coeff]]

			# # Normalized version
			x = [RGB[0] / (data1['l00'][i][0]) for i, RGB in enumerate(data1[coeff])]
			y = [RGB[0] / (data2['l00'][i][0]) for i, RGB in enumerate(data2[coeff])]

			# Plotting the scatter
			axes[axIndex(coeff)].scatter(x, y, label=rf'$Y_{coeffPedix(coeff)}$')

			# Handling diagonal line
			diagonalLine(coeff, min(x), max(x), axes)

		# Add labels and titles
		fig.suptitle(f'{filename}')
		axes[0].set_title('Zeroth order harmonics')
		axes[1].set_title('First order harmonics')
		axes[2].set_title('Second order harmonics')
		for i in range(3):
			axes[i].set_xlabel('Lighting coefficients')
			axes[i].set_ylabel('Lighting coefficients')
			axes[i].legend()
			axes[i].set_xlim([-0.5, 0.5])
			axes[i].set_ylim([-0.5, 0.5])

	plt.show()
