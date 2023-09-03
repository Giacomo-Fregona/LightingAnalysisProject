import matplotlib
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image
from point import point
import time


def medianFiltering(image, N=9):
	outputImage = ndimage.median_filter(image, size=(N, N, 1))
	return outputImage


pixelLength = 1024


class circle:
	def __init__(self, cx: int, cy: int, r: int, sigma=40, epsilon=0.2):  # constructor of the class
		self.center = point(cx, cy)
		self.r = r
		self.sigma = sigma
		self.epsilon = epsilon

		# Coefficients, one coordinate for each layer, using only the 0-th coordinate for grayscale images
		self.l00 = np.zeros(shape=(3), dtype=float)
		self.l1m1 = np.zeros(shape=3, dtype=float)
		self.l10 = np.zeros(shape=3, dtype=float)
		self.l11 = np.zeros(shape=(3), dtype=float)
		self.l2m2 = np.zeros(shape=(3), dtype=float)
		self.l2m1 = np.zeros(shape=(3), dtype=float)
		self.l20 = np.zeros(shape=(3), dtype=float)
		self.l21 = np.zeros(shape=(3), dtype=float)
		self.l22 = np.zeros(shape=(3), dtype=float)

	def __repr__(self):
		return f"Circle with center ({self.center.x}, {self.center.y}) and radius {self.r}."

	def __contains__(self, P: point):
		return (P.x - self.center.x) ** 2 + (P.y - self.center.y) ** 2 < self.r ** 2

	def normalAtPoint(self, P: point):  # Returns the normal vector in the point P of the sphere
		# if not P.belongsToCircle(self):
		#	raise Exception("The point does not belong to the circle.")
		# else:
		n = np.zeros(3, dtype=float)
		n[0] = P.x - self.center.x
		n[1] = P.y - self.center.y
		n[2] = np.sqrt(self.r ** 2 - (n[0]) ** 2 - (n[1]) ** 2)
		return n / np.linalg.norm(n)

	@staticmethod
	def Y00(n):
		return 1 / np.sqrt(4 * np.pi)

	@staticmethod
	def Y1m1(n):
		return np.sqrt(3 / (4 * np.pi)) * n[1]

	@staticmethod
	def Y10(n):
		return np.sqrt(3 / (4 * np.pi)) * n[2]

	@staticmethod
	def Y11(n):
		return np.sqrt(3 / (4 * np.pi)) * n[0]

	@staticmethod
	def Y2m2(n):
		return 3 * np.sqrt(5 / (12 * np.pi)) * n[0] * n[1]

	@staticmethod
	def Y2m1(n):
		return 3 * np.sqrt(5 / (12 * np.pi)) * n[1] * n[2]

	@staticmethod
	def Y20(n):
		return 0.5 * np.sqrt(5 / (4 * np.pi)) * (3 * (n[2] ** 2) - 1)

	@staticmethod
	def Y21(n):
		return 3 * np.sqrt(5 / (12 * np.pi)) * n[0] * n[2]

	@staticmethod
	def Y22(n):
		return 1.5 * np.sqrt(5 / (12 * np.pi)) * ((n[0] ** 2) - (n[1] ** 2))

	def renderedOnImage(self, image):
		""" Rendering the sphere on the image."""
		for x in range(pixelLength):
			firstFound = False  # Flag useful to speed up the algorithm
			for y in range(pixelLength):
				P = point(x, y)
				if P in self:
					firstFound = True
					n = self.normalAtPoint(P)

					match len(image.shape):
						case 3:  # RGB image
							for i in range(3):  # i cycling through the three RGB layers
								# n is the normal vector on the point P
								image[x, y, i] = self.l00[i] * np.pi * self.Y00(n) + \
												 self.l1m1[i] * (2 * np.pi / 3) * self.Y1m1(n) + \
												 self.l10[i] * (2 * np.pi / 3) * self.Y10(n) + \
												 self.l11[i] * (2 * np.pi / 3) * self.Y11(n) + \
												 self.l2m2[i] * (np.pi / 4) * self.Y2m2(n) + \
												 self.l2m1[i] * (np.pi / 4) * self.Y2m1(n) + \
												 self.l20[i] * (np.pi / 4) * self.Y20(n) + \
												 self.l21[i] * (np.pi / 4) * self.Y21(n) + \
												 self.l22[i] * (np.pi / 4) * self.Y22(n)
						case 2:  # Grayscale image
							image[x, y] = self.l00[0] * np.pi * self.Y00(n) + \
										  self.l1m1[0] * (2 * np.pi / 3) * self.Y1m1(n) + \
										  self.l10[0] * (2 * np.pi / 3) * self.Y10(n) + \
										  self.l11[0] * (2 * np.pi / 3) * self.Y11(n) + \
										  self.l2m2[0] * (np.pi / 4) * self.Y2m2(n) + \
										  self.l2m1[0] * (np.pi / 4) * self.Y2m1(n) + \
										  self.l20[0] * (np.pi / 4) * self.Y20(n) + \
										  self.l21[0] * (np.pi / 4) * self.Y21(n) + \
										  self.l22[0] * (np.pi / 4) * self.Y22(n)
						case other:
							raise Exception(
								"The image where to render the sphere has not the correct format of an RGB or grayscale image.")
				else:  # Since spheres are convex figures, we can skip some iterations
					if firstFound:
						break
		return image

	def fastRenderedOnImage(self, originalImage):
		""" Rendering ball on image faster using precomputation and iterative procedure for finding points on image"""
		image = originalImage.copy()

		# Precomputing mu, i.e. the fixed multipliers involved in each pixel's value estimation
		mu = np.zeros((9), dtype=float)

		# We can reduce the number of computation by observing some patterns
		# Note that this improvement is performed only once
		mu[0] = np.pi / np.sqrt(4 * np.pi)
		mu[1] = (2 * np.pi / 3) * np.sqrt(3 / (4 * np.pi))
		mu[2] = mu[1]
		mu[3] = mu[1]
		val = (np.pi / 4) * np.sqrt(5 / (12 * np.pi))
		mu[4] = 3 * val
		mu[5] = mu[4]
		mu[6] = 0.5 * np.sqrt(3) * val
		mu[7] = mu[4]
		mu[8] = 1.5 * val

		# Iterate on all the points in the down-right of the ball
		for xCoordinate in range(self.center.x - self.r + 1, self.center.x + 1):  # iterate on the radius
			d = int(np.floor(
				self.center.x - xCoordinate))  # distance from the point we are dealing with and the center (in x coordinate)
			c = int(np.floor(np.sqrt(self.r ** 2 - (
				d) ** 2)))  # distance from the center and the minimum y that is related to the xCoordinate
			addx = d  # distance between self.center.x and xCoordinate
			# print(f"d = {d}")
			for yCoordinate in range(self.center.y - c, self.center.y + 1):
				addy = int(np.floor(self.center.y - yCoordinate))  # distance between self.center.y and yCoordinate
				flag0 = point(xCoordinate, yCoordinate).isInImage()  # if the point is in the image
				flag1 = point(xCoordinate, yCoordinate + 2 * addy).isInImage()  # if the point is in the image
				flag2 = point(xCoordinate + 2 * addx, yCoordinate).isInImage()  # if the point is in the image
				flag3 = point(xCoordinate + 2 * addx,
							  yCoordinate + 2 * addy).isInImage()  # if the point is in the image
				# try:
				n = self.normalAtPoint(
					point(xCoordinate, yCoordinate))  # xCoordinate1 = xCoordinate, yCoordinate1 = yCoordinate
				n1 = np.array([n[0], -n[1], n[2]])  # xCoordinate1 = xCoordinate, yCoordinate1 = yCoordinate + 2*addy
				n2 = np.array([-n[0], n[1], n[2]])  # xCoordinate1 = xCoordinate + 2addx, yCoordinate = yCoordinate
				n3 = np.array(
					[-n[0], -n[1], n[2]])  # xCoordinate1 = xCoordinate + 2addx, yCoordinate = yCoordinate + 2*addy

				# Computing normal-dependent parameters
				Y = np.zeros((9), dtype=float)
				Y[0] = mu[0]
				Y[2] = mu[2] * n[2]
				Y[6] = mu[6] * (3 * (n[2] ** 2) - 1)
				Y[8] = mu[8] * ((n[0] ** 2) - (n[1] ** 2))

				Y[1] = mu[1] * n[1]
				Y[3] = mu[3] * n[0]
				Y[4] = mu[4] * n[0] * n[1]
				Y[5] = mu[5] * n[1] * n[2]
				Y[7] = mu[7] * n[0] * n[2]
				# Main idea is to note the fact that the normal vector of the ball are geometrically connected
				# So, we can save more or less 75% of the computation
				match len(image.shape):
					case 3:  # RGB image
						for i in range(3):  # i cycling through the three RGB layers
							val0123 = self.l00[i] * Y[0] + \
									  self.l10[i] * Y[2] + \
									  self.l20[i] * Y[6] + \
									  self.l22[i] * Y[8]
							val01 = self.l11[i] * Y[3] + \
									self.l21[i] * Y[7]
							val23 = -val01
							val03 = self.l2m2[i] * Y[4]
							val12 = -val03
							val02 = self.l1m1[i] * Y[1] + \
									self.l2m1[i] * Y[5]
							val13 = -val02

							if flag0: image[xCoordinate, yCoordinate, i] = val0123 + val01 + val03 + val02
							if flag1: image[xCoordinate, yCoordinate + 2 * addy, i] = val0123 + val01 + val12 + val13
							if flag2: image[xCoordinate + 2 * addx, yCoordinate, i] = val0123 + val23 + val12 + val02
							if flag3: image[
								xCoordinate + 2 * addx, yCoordinate + 2 * addy, i] = val0123 + val23 + val03 + val13

					case 2:  # Grayscale image
						val0123 = self.l00[0] * Y[0] + \
								  self.l10[0] * Y[2] + \
								  self.l20[0] * Y[6] + \
								  self.l22[0] * Y[8]
						val01 = self.l11[0] * Y[3] + \
								self.l21[0] * Y[7]
						val23 = -val01
						val03 = self.l2m2[0] * Y[4]
						val12 = -val03
						val02 = self.l1m1[0] * Y[1] + \
								self.l2m1[0] * Y[5]
						val13 = -val02

						if flag0: image[xCoordinate, yCoordinate] = val0123 + val01 + val03 + val02
						if flag1: image[xCoordinate, yCoordinate + 2 * addy] = val0123 + val01 + val12 + val13
						if flag2: image[xCoordinate + 2 * addx, yCoordinate] = val0123 + val23 + val12 + val02
						if flag3: image[
							xCoordinate + 2 * addx, yCoordinate + 2 * addy] = val0123 + val23 + val03 + val13
					case other:
						raise Exception(
							"The image where to render the sphere has not the correct format of an RGB or grayscale image.")
		return image

	def rendered(self):
		return self.fastRenderedOnImage(np.zeros(shape=(pixelLength, pixelLength, 3), dtype=np.uint8))

	def grayscaleRendered(self):
		return self.fastRenderedOnImage(np.zeros(shape=(pixelLength, pixelLength), dtype=np.uint8))

	def onImage(self, originalImage,
				width=2):  # return an RGB image with the grayscale original image in background and the circle guess in red

		image = originalImage.copy()

		if len(image.shape) == 2:  # grayscale image
			output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.short)
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					if image[i, j] == 255:  # adding points of the image
						for k in range(3):
							output[i, j, k] = 255
					if (abs(np.sqrt((i - self.center.x) ** 2 + (
							j - self.center.y) ** 2) - self.r) < width):  # adding points of the circle
						output[i, j, 0] = 0
						output[i, j, 1] = 255
						output[i, j, 2] = 0
		else:  # RGB image
			output = image
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					if (abs(np.sqrt((i - self.center.x) ** 2 + (
							j - self.center.y) ** 2) - self.r) < width):  # adding points of the circle
						output[i, j, 0] = 0
						output[i, j, 1] = 255
						output[i, j, 2] = 0
		return output

	def randomPoint(self, M):
		"Extracts a list of N points at random in the filling of the sphere."

		# Number of possible points
		maxNumber = int(
			np.floor(2 * self.r + 4 * np.sum([np.sqrt(self.r ** 2 - i ** 2) for i in range(1, int(np.floor(self.r)))])))

		# Points list
		pointsList = []

		for _ in range(M):
			flag = True
			while flag:  # Some points may be outside the image borders
				# Select at random the "index" of the point
				index = np.random.randint(0, maxNumber)

				# Compute the bijection and find the point
				s = 0  # Actual sum
				for xIndex in range(int(np.floor(-self.r)) + 1, int(np.floor(self.r))):
					l = 2 * int(np.floor(np.sqrt(int(np.floor(self.r)) ** 2 - xIndex ** 2)))
					s += l
					if s > index:
						s -= l
						yIndex = index - s - int(np.floor(l / 2))
						if (self.center.x + xIndex in range(0, pixelLength)) and (
								self.center.y + yIndex in range(0, pixelLength)):
							pointsList.append(point(x=self.center.x + xIndex, y=self.center.y + yIndex))
							flag = False
						else:
							pass  # In that case we must repeat another while iteration because the point is not inside the image borders
						break

		return pointsList

	def extimateCoefficients(self, image, M=9, median=True):
		"""Extimate the rendering coefficients of the sphere using N points."""

		# Constructiong the list of the points
		pointsList = self.randomPoint(M)

		# Constructing the matrix A
		A = []
		for p in pointsList:
			n = self.normalAtPoint(p)
			A.append([np.pi * self.Y00(n), (2 * np.pi / 3) * self.Y1m1(n), (2 * np.pi / 3) * self.Y10(n),
					  (2 * np.pi / 3) * self.Y11(n), (np.pi / 4) * self.Y2m2(n), (np.pi / 4) * self.Y2m1(n),
					  (np.pi / 4) * self.Y20(n), (np.pi / 4) * self.Y21(n), (np.pi / 4) * self.Y22(n)]
					 )
		A = np.array(A)

		match len(image.shape):
			case 3:  # RGB image
				for i in range(3):  # i cycling through the three RGB layers

					# Constructing the vector b for each layer
					b = np.array([image[p.x, p.y, i] for p in pointsList])

					# Solving the system
					l = np.linalg.lstsq(A, b, rcond=None)[0]

					# Storing the results
					self.l00[i] = l[0]
					self.l1m1[i] = l[1]
					self.l10[i] = l[2]
					self.l11[i] = l[3]
					self.l2m2[i] = l[4]
					self.l2m1[i] = l[5]
					self.l20[i] = l[6]
					self.l21[i] = l[7]
					self.l22[i] = l[8]

			case 2:  # Grayscale image

				# Constructing the vector b
				b = np.array([image[p.x, p.y] for p in pointsList])

				# Solving the system
				l = np.linalg.lstsq(A, b, rcond=None)[0]

				# Storing the results
				self.l00[0] = l[0]
				self.l1m1[0] = l[1]
				self.l10[0] = l[2]
				self.l11[0] = l[3]
				self.l2m2[0] = l[4]
				self.l2m1[0] = l[5]
				self.l20[0] = l[6]
				self.l21[0] = l[7]
				self.l22[0] = l[8]

			case other:
				raise Exception(
					"The image where to render the sphere has not the correct format of an RGB or grayscale image.")


if __name__ == '__main__':

	##### LIGTHING ANALYSIS DEMO #####

	# Let's open a RGB image from our sample folder
	imageName = "./Samples/DALLE2/DALLE2_1.png"
	originalImage = np.asarray(Image.open(imageName), dtype=np.uint8)
	print(f'Image "{imageName}" opened.')

	# We have already estimated for you the sphere's center and radious
	C = circle(587, 432, 301)

	# Let's estimate the l_{i,j} coefficients using M = 150 points
	C.extimateCoefficients(originalImage, M=150)
	print("The coefficients have been estimated.")

	# We can now render the sphere on the image. We measure the computation time
	start = time.time()
	rendered = C.fastRenderedOnImage(originalImage)
	end = time.time()
	print(f"Image rendered in {end - start} s.")

	# Finally we see the results of our computation. In the two subplots of the figure we represent:
	# 	- the original image with the starting circle guess highlighted in smart green
	# 	- the image with rendered sphere constructed from the estimated coefficients

	matplotlib.rcParams['figure.figsize'] = [14, 7]

	# Subplot 1
	plt.subplot(121)
	plt.title("Original image with spheres circle")
	plt.imshow(C.onImage(originalImage))

	# Subplot 2
	plt.subplot(122)
	plt.title('Rendered sphere')
	plt.imshow(rendered)

	plt.show()
