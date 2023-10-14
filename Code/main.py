import matplotlib
import numpy as np

from expectation_maximization import EM
from interaction_lib import interactiveGuess
from PIL import Image

import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from Code.archive import Archive

##### SAMPLE USAGE OF THE REPOSITORY FUNCTIONS #####

# Opening an image from sample folder
imageName = "./Samples/prompt/prompt_2.png"
originalImage = np.asarray(Image.open(imageName), dtype=np.uint8)
print(f'Image "{imageName}" opened.')

# Calling interactive method to define the first guess
C = interactiveGuess(originalImage)

# Refining the guess with expectation-maximization procedure
C = EM(C=C, rounds=10, visual=0, finalVisual=0, erase=1)

# Estimating the coefficients
C.estimateCoefficients(M=1000)

# Getting the estimated coefficients for each RGB layer
coefficients = np.array(
	[[C.l00[i], C.l1m1[i], C.l10[i], C.l11[i], C.l2m2[i], C.l2m1[i], C.l20[i], C.l21[i], C.l22[i]] for i in range(3)])
print("\nEstimated coefficients:")
print(coefficients)

# Rendering the sphere and showing it alongside the original image showing the circle guess
rendered = C.fastRenderedOnImage(originalImage)

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

""" Adding the result of computation to the archive """

# Adding image_id attribute to the circle
C.image_id = imageName

# Adding circle to archive
pa: Archive = Archive.load(Archive.PROMPT)
pa.append(C)
pa.save()
