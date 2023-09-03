import matplotlib
import numpy as np

from expectation_maximization import EM
from interaction_lib import interactiveGuess
from PIL import Image

matplotlib.use('TkAgg')

##### SAMPLE USAGE OF THE REPOSITORY FUNCTIONS #####

# Opening an image from sample folder
imageName = "./Samples/DALLE2/DALLE2_1.png"
originalImage = np.asarray(Image.open(imageName), dtype=np.uint8)
print(f'Image "{imageName}" opened.')

# Calling interactive method to define the first guess
C = interactiveGuess(originalImage)

# Refining the guess with expectation-maximization procedure
C = EM(originalImage, C, rounds=10, visual=0, finalVisual=0, erase=1)

# Estimating the coefficients
C.extimateCoefficients(originalImage, M=250)

# Getting the estimated coefficients for each RGB layer
coefficients = np.array([[C.l00[i], C.l1m1[i], C.l10[i], C.l11[i], C.l2m2[i], C.l2m1[i], C.l20[i], C.l21[i], C.l22[i]] for i in range(3)])
print("\nEstimated coefficients:")
print(coefficients)
