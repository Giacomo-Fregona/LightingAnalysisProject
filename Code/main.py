import matplotlib
import numpy as np
import os
import threading

from expectation_maximization import EM
from interaction_lib import interactiveGuess
from PIL import Image

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

##### SAMPLE USAGE OF THE REPOSITORY FUNCTIONS #####

# Opening an image from sample folder
flag = ""
while True:
	flag = input("real, prompt or dalle2FromReal? ").lower().strip()
	if flag == "real" or flag == "prompt" or flag == "dalle2FromReal":
		break
	else:
		print("Error: Answer must be real or dalle2")

where = "./Samples/"+flag
png_images = [f for f in os.listdir(where) if f.endswith(".png")]

number = -1
while True:
	for i, my_images in enumerate(png_images):
		print(f"{i}: {my_images}")
	number = input("Choose one number in: ")
	try:
		number = int(number)
		if 0 <= number < len(png_images):
			break
		else:
			print("Error: Answer must be between 0 and " + str(len(png_images)))
	except ValueError:
		print("Invalid input. Please enter a number")

imageName = where + "/" + flag + "_" + str(number) + ".png"
print(imageName)

originalImage = np.asarray(Image.open(imageName), dtype=np.uint8)
print(f'Image "{imageName}" opened.')

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

for i in range(numberOfSpheres):
	C = allMyC[i]
	# Refining the guess with expectation-maximization procedure
	C = EM(originalImage, C, rounds=10, visual=0, finalVisual=0, erase=1)

	# Estimating the coefficients
	C.extimateCoefficients(originalImage, M=1000)

	# Getting the estimated coefficients for each RGB layer
	coefficients = np.array([[C.l00[i], C.l1m1[i], C.l10[i], C.l11[i], C.l2m2[i], C.l2m1[i], C.l20[i], C.l21[i], C.l22[i]] for i in range(3)])
	print("\nEstimated coefficients:")
	print(coefficients)

	# Rendering the sphere and showing it alongside the original image showing the circle guess
	rendered = C.fastRenderedOnImage(originalImage)
	allMyRendered[i] = rendered

	matplotlib.rcParams['figure.figsize'] = [20, 7]

	# Subplot 1
	plt.subplot(131)
	plt.title("Original image with spheres circle")
	plt.imshow(C.onImage(originalImage))

	# Subplot 2
	plt.subplot(132)
	plt.title('Median filtered sphere')
	plt.imshow(C.medianFiltered(originalImage))

	# Subplot 3
	plt.subplot(133)
	plt.title('Rendered sphere')
	plt.imshow(rendered)

	plt.show()


for i in range(numberOfSpheres):
	for x in range(allMyC[i].center.x - allMyC[i].r, allMyC[i].center.x + allMyC[i].r + 1):
		for y in range(allMyC[i].center.y - allMyC[i].r, allMyC[i].center.y + allMyC[i].r + 1):
			for layer in range(3):
				originalImage[x, y, layer] = np.clip(allMyRendered[i][x, y, layer], 0, 255)


# Display the modified image
plt.figure()
plt.title('Modified Original Image')
plt.imshow(originalImage)
plt.show()

""" Adding the result of computation to the archive """

# Adding image_id attribute to the circle
C.image_id = imageName

# Adding circle to archive
pa: Archive = Archive.load(Archive.PROMPT)
pa.append(C)
pa.save()
