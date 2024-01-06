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

	def __init__(self, fileName: str):
		self.fileName = fileName
		super().__init__()

	def __repr__(self):

		selfDict = self.asDict()
		output = f'\nArchive {self.fileName} containing {len(selfDict.keys())} images.'
		for imageId in selfDict.keys():
			output += f'\n\t{imageId} --> {len(selfDict[imageId])} circles'
		return output

	@staticmethod
	def load(whichArchive):

		# Retrieving the archive
		with open(whichArchive, 'rb') as file:
			archive = pickle.load(file)
		archive.fileName = whichArchive

		# Loading images in each circle
		C: circle
		toRemove = []
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
				toRemove.append(i)

		toRemove.reverse()

		for i in toRemove:
			archive.pop(i)

		return archive

	def save(self):

		# Removing all the images to save data space
		for C in self:
			C.image = None

		# Dumping
		with open(self.fileName, 'wb') as file:
			return pickle.dump(self, file)

	def asDict(self):
		"""
		archive returned as a dictionary where keys are image_ids and values are lists of corresponding circles
		@return: the dictionary
		"""

		outputDict = {}

		C: circle
		for C in self:
			if not (C.image_id in outputDict):
				# Handling the case the image id is not already present as key
				outputDict[C.image_id] = [C]

			else:
				# Handling the case the image id is already present as key
				outputDict[C.image_id].append(C)

		return outputDict

	def visualize(self, imageId: str = None, archiveIndex: int = None):

		if (imageId is None) and (archiveIndex is None):
			# Loading the entire archive as dict
			workingDict = self.asDict()
		else:
			# Working on a reduced archive
			if not (archiveIndex is None):
				imageId = self[archiveIndex].image_id
			workingDict = {imageId: self.asDict()[imageId]}

		print('Dict ready!')

		visualizationDict = {}

		for i, circles in enumerate(workingDict.values()):

			# Retrieving image data
			originalImage = circles[0].image
			image = originalImage.copy()
			imageId = circles[0].image_id
			print(f'\nProcessing image {imageId} ({i + 1}/{len(workingDict.values())})')

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
			visualizationDict[imageId] = [originalImage, image]

		print(f'\nImage rendering finished.')

		# Visualization
		for imageId in visualizationDict.keys():
			matplotlib.rcParams['figure.figsize'] = [20, 7]

			# Subplot 1: original image
			plt.subplot(121)
			plt.title(f"{imageId}")
			plt.imshow(visualizationDict[imageId][0])

			# Subplot 2: rendered image
			plt.subplot(122)
			plt.title("Index [Archive Index]")
			plt.imshow(visualizationDict[imageId][1])

			plt.show()





if __name__ == '__main__':

	# Loading and visualizing the content of the real photos archive
	realArchive: Archive = Archive.load(Archive.REAL)
	print(realArchive)
	realArchive.visualize()
