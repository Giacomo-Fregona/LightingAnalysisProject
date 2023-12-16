import matplotlib
import numpy as np
import os
import threading

from Code.expectation_maximization import EM
from Code.interaction_lib import interactiveGuess
from PIL import Image
from Code.archive import Archive

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

##### SAMPLE USAGE OF THE REPOSITORY FUNCTIONS #####

# Opening an image from sample folder
flag = ""
while True:
	flag = input("real, prompt or variation? ").lower().strip()
	if flag == "real" or flag == "prompt" or flag == "variation":
		break
	else:
		print("Error: Answer must be real or dalle2")

where = "./Samples/"+flag
png_images = [f for f in os.listdir(where) if f.endswith(".png")]

d = {}

for im in png_images:
	imageName = where + "/" + im

	originalImage = np.asarray(Image.open(imageName), dtype=np.uint8)

	def get_number_of_spheres():
		global numberOfSpheres
		while True:
			input_str = input("Choose the number of spheres you want to guess: ")
			try:
				numberOfSpheres = int(input_str)
				if numberOfSpheres > 0:
					break
				else:
					print("Error: Answer must be higher than 0")
			except ValueError:
				print("Invalid input. Please enter a number")

	# Create a separate thread for user input
	input_thread = threading.Thread(target=get_number_of_spheres)
	input_thread.daemon = True
	input_thread.start()

	plt.imshow(originalImage)
	plt.title("How many spheres you want to guess? \n Write it in terminal, then close this window", fontsize=16)
	plt.show()

	# Wait for the user to provide the number of spheres
	input_thread.join()

	allMyC = [None] * numberOfSpheres
	allMyRendered = [None] * numberOfSpheres


	for i in range(numberOfSpheres):
		# Calling interactive method to define the first guess
		allMyC[i] = interactiveGuess(originalImage)

	d[im] = allMyC, allMyRendered, numberOfSpheres

for im in png_images:
	imageName = where + "/" + im

	originalImage = np.asarray(Image.open(imageName), dtype=np.uint8)

	allMyC, allMyRendered, numberOfSpheres = d[im]

	for i in range(numberOfSpheres):

		# Refining the guess with expectation-maximization procedure
		C = EM(originalImage, allMyC[i], rounds=10, visual=0, finalVisual=0, erase=0)

		# Estimating the coefficients
		C.estimateCoefficients(originalImage, M=1000)

		# Getting the estimated coefficients for each RGB layer
		coefficients = np.array([[C.l00[i], C.l1m1[i], C.l10[i], C.l11[i], C.l2m2[i], C.l2m1[i], C.l20[i], C.l21[i], C.l22[i]] for i in range(3)])
		print("\nEstimated coefficients:")
		print(coefficients)

		# # Rendering the sphere and showing it alongside the original image showing the circle guess
		# rendered = C.fastRenderedOnImage(originalImage)
		# allMyRendered[i] = rendered

		""" Adding the result of computation to the archive """

		# Adding image_id attribute to the circle
		C.image_id = imageName

		# Adding circle to archive
		if (flag == "real"):
			pa: Archive = Archive.load(Archive.REAL)
		elif (flag == "prompt"):
			pa: Archive = Archive.load(Archive.PROMPT)
		elif (flag == "variation"):
			pa: Archive = Archive.load(Archive.VARIATION)
		pa.append(C)
		pa.save()
#
# for im in png_images:
# 	imageName = where + "/" + im
#
# 	originalImage = np.asarray(Image.open(imageName), dtype=np.uint8)
#
# 	allMyC, allMyRendered, numberOfSpheres = d[im]
#
# 	for i in range(numberOfSpheres):
# 		matplotlib.rcParams['figure.figsize'] = [20, 7]
#
# 		# Subplot 1
# 		plt.subplot(131)
# 		plt.title("Original image with spheres circle")
# 		plt.imshow(C.onImage(originalImage))
#
# 		# Subplot 2
# 		plt.subplot(132)
# 		plt.title('Median filtered sphere')
# 		plt.imshow(C.medianFiltered(originalImage))
#
# 		# Subplot 3
# 		plt.subplot(133)
# 		plt.title('Rendered sphere')
# 		plt.imshow(rendered)
#
# 		plt.show()
#
# 		if(False): #to use for presentation
# 			pixelLength = 1024
# 			for i in range(numberOfSpheres):
# 				for x in range(min(allMyC[i].center.x - allMyC[i].r, pixelLength), min(allMyC[i].center.x + allMyC[i].r + 1, pixelLength)):
# 					for y in range(min(allMyC[i].center.y - allMyC[i].r, pixelLength), min(allMyC[i].center.y + allMyC[i].r + 1, pixelLength)):
# 						for layer in range(3):
# 							originalImage[x, y, layer] = np.clip(allMyRendered[i][x, y, layer], 0, 255)
#
#
# 			# Display the modified image
# 			plt.figure()
# 			plt.title('Modified Original Image')
# 			plt.imshow(originalImage)
# 			plt.show()
#
# 		""" Adding the result of computation to the archive """
#
# 		# Adding image_id attribute to the circle
# 		C.image_id = imageName
#
# 		# Adding circle to archive
# 		if (flag == "real"):
# 			pa: Archive = Archive.load(Archive.REAL)
# 		elif (flag == "prompt"):
# 			pa: Archive = Archive.load(Archive.PROMPT)
# 		elif (flag == "variation"):
# 			pa: Archive = Archive.load(Archive.VARIATION)
# 		pa.append({imageName: C})
# 		pa.save()
