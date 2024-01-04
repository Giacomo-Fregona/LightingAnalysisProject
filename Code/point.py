import numpy as np

pixelLength = 1024


class point:
	def __init__(self, x: int, y: int):
		self.x = x
		self.y = y

	def isInImage(self):
		return not ((self.x < 0) or (self.x >= pixelLength) or (self.y < 0) or (self.y >= pixelLength))

	def belongsToCircle(self, C):
		return (self.x - C.center.x) ** 2 + (self.y - C.center.y) ** 2 < C.r ** 2

	def __repr__(self):
		return f"({self.x}, {self.y})"

	def onImage(self, image,
				width=10):  # return an RGB image with the grayscale original image in background and the circle guess in red
		if len(image.shape) == 2:  # grayscale image
			output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.short)
			for i in range(image.shape[0]):
				for j in range(image.shape[1]):
					if image[i, j] == 255:  # adding points of the image
						for k in range(3):
							output[i, j, k] = 255

			# Adding the point
			rad = int(np.ceil(width / 2))
			for i in range(-rad, rad + 1):
				for j in range(-rad, rad + 1):
					output[self.x + i, self.y + j, 0] = 0
					output[self.x + i, self.y + j, 1] = 255
					output[self.x + i, self.y + j, 2] = 0
		else:  # RGB image
			output = image

			# Adding the point
			rad = int(np.ceil(width / 2))
			for i in range(-rad, rad + 1):
				for j in range(-rad, rad + 1):
					output[self.x + i, self.y + j, 0] = 0
					output[self.x + i, self.y + j, 1] = 255
					output[self.x + i, self.y + j, 2] = 0

		return output

	@staticmethod
	def collectionOnImage(pointsList, image=np.zeros((pixelLength, pixelLength, 3))):
		"""Static method that prints many points on an image."""
		for i in range(len(pointsList)):
			image = pointsList[i].onImage(image)
		return image
