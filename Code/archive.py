"""
File containing methods for saving and retrieving data coming from the lighting analysis
"""
import pickle
import matplotlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

from circle import circle

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

	def __repr__(self):

		self_dict = self.as_dict()
		output = f'\nArchive {self.file_name} containing {len(self_dict.keys())} images.'
		for image_id in self_dict.keys():
			output += f'\n\t{image_id} --> {len(self_dict[image_id])} circles'
		return output

	@staticmethod
	def load(which_archive):

		# Retrieving the archive
		with open(which_archive, 'rb') as file:
			archive = pickle.load(file)
		archive.file_name = which_archive

		# Loading images in each circle
		C: circle
		to_remove = []
		for i in range(len(archive)):
			C = archive[i]
			try:
				C.image = np.asarray(Image.open(C.image_id), dtype=np.uint8)
			except:

				# Handling case image not in the /Samples database
				if C.image is not None:
					plt.title(f'IMAGE NOT FOUND IN SAMPLES FOLDER: {C.image_id}')
					plt.imshow(C.image)
					plt.show()
				print(
					f'An error occurred loading image {C.image_id}. Do not save the archive if you do not want to lose the corresponding data')
				to_remove.append(i)

		to_remove.reverse()

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

	def as_dict(self):
		"""
		archive returned as a dictionary where keys are image_ids and values are lists of corresponding circles
		@return: the dictionary
		"""

		output_dict = {}

		C: circle
		for C in self:
			if not (C.image_id in output_dict):
				# Handling the case the image id is not already present as key
				output_dict[C.image_id] = [C]

			else:
				# Handling the case the image id is already present as key
				output_dict[C.image_id].append(C)

		return output_dict

	def visualize(self, image_id: str = None, archive_index: int = None):

		if (image_id is None) and (archive_index is None):
			# Loading the entire archive as dict
			working_dict = self.as_dict()
		else:
			# Working on a reduced archive
			if not (archive_index is None):
				image_id = self[archive_index].image_id
			working_dict = {image_id: self.as_dict()[image_id]}

		print('Dict ready!')

		visualization_dict = {}

		for i, circles in enumerate(working_dict.values()):

			# Retrieving image data
			original_image = circles[0].image
			image = original_image.copy()
			image_id = circles[0].image_id
			print(f'\nProcessing image {image_id} ({i + 1}/{len(working_dict.values())})')

			# Adding circles to image
			C: circle
			for C in circles:
				# Rendering
				image = C.fastRenderedOnImage(image)

				# Adding indexes
				cv2.putText(image, f'{circles.index(C)}[{self.index(C)}]', (C.center.y, C.center.x),
							cv2.FONT_HERSHEY_PLAIN, 4,
							(255, 255, 255), 2)

				print(f'\tRendered {circles.index(C) + 1}/{len(circles)}.')

			# Storing resulting images
			visualization_dict[image_id] = [original_image, image]

		print(f'\nImage rendering finished.')

		# Visualization
		for image_id in visualization_dict.keys():
			matplotlib.rcParams['figure.figsize'] = [20, 7]

			# Subplot 1: original image
			plt.subplot(121)
			plt.title(f"{image_id}")
			plt.imshow(visualization_dict[image_id][0])

			# Subplot 2: rendered image
			plt.subplot(122)
			plt.title("Index [Archive Index]")
			plt.imshow(visualization_dict[image_id][1])

			plt.show()





if __name__ == '__main__':
	# Archive bootstrapping
	real_archive = Archive(Archive.REAL)
	real_archive.save()

	prompt_archive = Archive(Archive.PROMPT)
	prompt_archive.save()

	variation_archive = Archive(Archive.VARIATION)
	variation_archive.save()
