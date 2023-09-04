import matplotlib
import numpy as np

from expectation_maximization import EM
from interaction_lib import interactiveGuess
from PIL import Image
from circle import circle

matplotlib.use('TkAgg')

##### SAMPLE USAGE OF THE REPOSITORY FUNCTIONS #####

# Opening an image from sample folder
real = False
if real:
	imageName = "./Samples/real/real_10.png"
	where = "./Code/numberOfPointsForRendering/RealImage"
	C = circle(479, 515, 201)
else:
	imageName = "./Samples/dalle2/dalle2_1.png"
	where = "./Code/numberOfPointsForRendering/DALLE2Image"
	C = circle(587, 432, 301)

originalImage = np.asarray(Image.open(imageName), dtype=np.uint8)
print(f'Image "{imageName}" opened.')


# Calling interactive method to define the first guess
counterOfPoint = 0
for xCoordinate in range(C.center.x - C.r + 1, C.center.x + C.r):
	d = int(np.floor(C.center.x - xCoordinate))  # distance from the point we are dealing with and the center (in x coordinate)
	c = int(np.floor(np.sqrt(C.r ** 2 - (d) ** 2)))  # distance from the center and the minimum y that is related to the xCoordinate
	for yCoordinate in range(C.center.y - c, C.center.y + c):
		counterOfPoint = counterOfPoint + 1

setOfM = [10, 200, 500, 1000, counterOfPoint*(5/100), counterOfPoint*(10/100)]

test = 10

for M in setOfM:
	print(M)
	M = int(np.floor(M))

	setOfCoefficients = np.zeros((test, 3, 9))


	for i in range(0,test):
		print(i)
		if real:
			C = circle(587, 432, 301)
		else :
			C = circle(479, 515, 201)
		# Estimating the coefficients
		C.extimateCoefficients(originalImage, M = M)
		# Getting the estimated coefficients for each RGB layer
		coefficients = np.array([[C.l00[i], C.l1m1[i], C.l10[i], C.l11[i], C.l2m2[i], C.l2m1[i], C.l20[i], C.l21[i], C.l22[i]] for i in range(3)])
		setOfCoefficients[i] = coefficients

	#print(setOfCoefficients)
	varianceOfCoeffients = np.var(setOfCoefficients, axis = 0)
	minVar = np.min(varianceOfCoeffients)
	maxVar = np.max(varianceOfCoeffients)
	meanOfCoefficients = np.mean(setOfCoefficients, axis = 0)
	minMean = np.min(meanOfCoefficients)
	maxMean = np.max(meanOfCoefficients)
	
	with open(f"./{where}/coefficients{M}.txt", "a") as f:
		f.write("M = " + str(M) + "\n")
		
		f.write("variance: \n")
		for value in varianceOfCoeffients:
			f.write(f"{np.floor(value)}\n")
		f.write("min = " + str(int(np.floor(minVar))) + "\n")
		f.write("max = " + str(int(np.floor(maxVar))) + "\n")
		f.write("\n\n")
		
		f.write("mean: \n")
		for value in meanOfCoefficients:
			f.write(f"{np.floor(value)}\n")
		f.write("min = " + str(int(np.floor(minMean))) + "\n")
		f.write("max = " + str(int(np.floor(maxMean))) + "\n")
		f.write("\n\n")
		
		for value in setOfCoefficients:
			f.write(f"{value}\n")
		f.write("\n\n")







