"""
File containing methods for saving and retrieving data coming from the lighting analysis
"""

import pickle
from Code.circle import circle

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class Archive(list):
	REAL = './Archive/real.pkl'
	PROMPT = './Archive/prompt.pkl'
	VARIATION = './Archive/variation.pkl'
	REAL_DARIO = './ArchiveDario/real.pkl'
	PROMPT_DARIO = './ArchiveDario/prompt.pkl'
	VARIATION_DARIO = './ArchiveDario/variation.pkl'
	REAL_GIACOMO = './ArchiveGiacomo/real.pkl'
	PROMPT_GIACOMO = './ArchiveGiacomo/prompt.pkl'
	VARIATION_GIACOMO = './ArchiveGiacomo/variation.pkl'

	def __init__(self, file_name: str):
		self.file_name = file_name
		super().__init__()

	@staticmethod
	def load(which_archive):

		# Retrieving the archive
		with open(which_archive, 'rb') as file:
			archive = pickle.load(file)

		archive.file_name = which_archive

		print(len(archive))

		# Loading images in each circle
		C: circle
		to_remove = []
		for i in range(len(archive)):
			C = archive[i]
			try:
				C.image = np.asarray(Image.open(C.image_id), dtype=np.uint8)
			except:

				#Handling case image not in the /Samples database
				if C.image is not None:
					plt.title(f'IMAGE NOT FOUND IN SAMPLES FOLDER: {C.image_id}')
					plt.imshow(C.image)
					plt.show()
				print(f'An error occurred loading image {C.image_id}. Do not save the archive if you do not want to lose the corresponding data')
				to_remove.append(i)

		to_remove.reverse()
		print(to_remove)
		for i in to_remove:
			archive.pop(i)

		return archive

	def save(self):

		# Removing all the images to save data space
		for C in self:
			C.image = None

		# Dumping
		with open(self.file_name, 'wb') as file:
			return pickle.dump(self, file)


if __name__ == '__main__':
	# Archive bootstrapping
	real_archive = Archive(Archive.REAL)
	real_archive.save()

	prompt_archive = Archive(Archive.PROMPT)
	prompt_archive.save()

	variation_archive = Archive(Archive.VARIATION)
	variation_archive.save()
