import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math
from circle import circle

matplotlib.use('TkAgg')

# ERASER

class eraser:
	def __init__(self, image, alwaysRefresh=1):

		# Defining plot objects
		self.fig, self.ax = plt.subplots()
		self.imsh = None
		self.image = image

		# Circle (eraser appearance)
		self.circle = matplotlib.patches.Circle(xy=(512, 512), radius=30, color='yellow',
												fill=False, alpha=0.4)
		self.ax.add_patch(self.circle)

		# Mouse event variables
		self.press = False
		self.position = None

		# Speed up or not
		self.alwaysRefresh = alwaysRefresh

	def connect(self):
		"""Connect to all the events we need."""
		self.cidpress = self.circle.figure.canvas.mpl_connect(
			'button_press_event', self.on_press)
		self.cidrelease = self.circle.figure.canvas.mpl_connect(
			'button_release_event', self.on_release)
		self.cidmotion = self.circle.figure.canvas.mpl_connect(
			'motion_notify_event', self.on_motion)

	def show(self):
		"""Intruducing the slider for the radius and showing the interface."""
		self.imsh = plt.imshow(self.image, cmap='gray', vmin=0, vmax=255)

		# Make a vertically oriented slider to control the radius
		axSlider = self.fig.add_axes([0.1, 0.15, 0.0225, 0.63])
		radiusSlider = Slider(
			ax=axSlider,
			label="Radius width",
			valmin=10,
			valmax=150,
			valinit=40,
			orientation="vertical"
		)

		# The function to be called anytime a slider's value changes
		def update(val):
			self.circle.set_radius(val)  # Update radius value
			self.circle.figure.canvas.draw()  # Refresh the image

		# Register the update function
		radiusSlider.on_changed(update)

		# Showing the interface
		self.fig.suptitle('Please erase noise and then close this window.', fontsize=16)
		plt.show()

	def on_press(self, event):
		"""Start erasing at mouse click"""

		if event.inaxes != self.circle.axes or not (event.inaxes == self.ax):
			return

		# Update press variable
		self.press = True

		# Update the position of the event
		self.position = (event.xdata, event.ydata)

		# Move and fill the circle
		self.circle.set(fill=True, center=(event.xdata, event.ydata))
		self.circle.figure.canvas.draw()

		# Drawing black pixels on the image cycling through the x and y coordinates of the points of the ball
		for i in range(int(max(0, self.position[0] - self.circle.get_radius())),
					   int(min(self.image.shape[0], self.position[0] + self.circle.get_radius()))):
			value = np.floor(
				np.sqrt(max(0, self.circle.get_radius() ** 2 - (i - self.position[0] - 1) ** 2)))  # Pitagoras triangle
			m = int(max(0, self.position[1] - value))  # x inf of the ball for coordinate i
			M = int(min(self.image.shape[0], self.position[1] + value))  # x sup of the ball for coordinate i
			self.image[m: M, i] = 0  # Adding the 0 pixels

			# Refreshing the image
			if self.alwaysRefresh:
				self.imsh.set_data(self.image)
				plt.draw()

	def on_motion(self, event):
		"""Update the rubber appearance during motion and continue erasing if the mouse button is pressed"""
		if self.position is None or event.inaxes != self.circle.axes or not (event.inaxes == self.ax):
			return

		# Update the position of the event
		self.position = (event.xdata, event.ydata)
		self.circle.set(center=self.position)
		self.circle.figure.canvas.draw()

		# Continue erasing
		if self.press:
			"""Same code of on_press"""

			# Drawing black pixels on the image cycling through the x and y coordinates of the points of the ball
			for i in range(int(max(0, self.position[0] - self.circle.get_radius())),
						   int(min(self.image.shape[0], self.position[0] + self.circle.get_radius()))):
				value = np.floor(np.sqrt(
					max(0, self.circle.get_radius() ** 2 - (i - self.position[0] - 1) ** 2)))  # Pitagoras triangle
				m = int(max(0, self.position[1] - value))  # x inf of the ball for coordinate i
				M = int(min(self.image.shape[0], self.position[1] + value))  # x sup of the ball for coordinate i
				self.image[m: M, i] = 0  # Adding the 0 pixels
			# print("ONPRESS");

			# Refreshing the image
			if self.alwaysRefresh:
				self.imsh.set_data(self.image)
				plt.draw()

	def on_release(self, event):
		"""Return to initial variable configuration"""
		self.press = False
		self.circle.set(fill=False)
		self.circle.figure.canvas.draw()

		# Refreshing the image in the case this has never happened before (fast mode)
		if not self.alwaysRefresh:
			self.imsh.set_data(self.image)
			plt.draw()

	def close(self):
		"""Disconnect all callbacks and return the resulting image."""
		self.circle.figure.canvas.mpl_disconnect(self.cidpress)
		self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
		self.circle.figure.canvas.mpl_disconnect(self.cidmotion)
		return self.image

# GUESS
def tellme(s):
	plt.title(s, fontsize=16)
	plt.draw()


def interactiveGuess(image,
					 max_y_difference=20):  # guessing 3 points in the border of the circle. max_y_difference is the maximal acceptable difference between two y coordinates in order to find the straight lines intersection

	print("\n --- START OF INTERACTIVE GUESS --- ")

	plt.imshow(image)

	fig = plt.gcf()
	ax = fig.gca()

	tellme('You will define a circle, click to begin.')

	plt.waitforbuttonpress()

	while True:

		pts = []
		while len(pts) < 3 or len(intersection) < 1:

			print("\nSelecting three points on the border...")
			tellme('Please, select three points on the border.')
			pts = np.asarray(plt.ginput(3, timeout=-1))  # Taking the points as input
			print("The selected points are: ", pts[0], pts[1], pts[2], ".")

			print("Finding the intersection...")
			intersection = []  # Here we will store the intersections between the two lines defined by couples of points (pts[0], pts[1]) and (pts[1], pts[2])
			if (abs(pts[1][1] - pts[0][1]) > 1 / 100) and (abs(pts[2][1] - pts[1][
				1]) > 1 / 100):  # We avoid the case (pts[0], pts[1]) and (pts[1], pts[2]) are vertical. This should happen with probability 0

				for x in range(image.shape[0]):  # Cycling through the x coordinate of the image

					# y point coordinate in the first line
					y_1 = -(pts[1][0] - pts[0][0]) / (pts[1][1] - pts[0][1]) * (x - (pts[1][0] + pts[0][0]) / 2) + (
							pts[1][1] + pts[0][1]) / 2

					# y point coordinate in the second line
					y_2 = -(pts[2][0] - pts[1][0]) / (pts[2][1] - pts[1][1]) * (x - (pts[2][0] + pts[1][0]) / 2) + (
							pts[2][1] + pts[1][1]) / 2

					if abs(y_1 - y_2) < max_y_difference:  # Selecting the points if they are close to each other (less than max_y_difference)
						intersection.append([x, y_1])
						intersection.append([x, y_2])

			print("The intersection is: \n", np.array(intersection))
			if len(intersection) < 1:
				print("Empty intersection found.")
				tellme('Not able to find points in the intersection. Restarting...')
				pts = []
				time.sleep(1)  # Wait a second
				continue

			if len(pts) < 3:
				print("Selected points problem.")
				tellme('Not enough points selected. Restarting...')
				pts = []
				time.sleep(1)  # Wait a second

		print("\nComputing the circle coordinates and radius...")
		center = np.mean(intersection,
						 axis=0)  # Averaging the coordinate values found in the intersection in order to discover the center

		r = math.dist(pts[0], (center[0], center[1]))  # Computing our guess for the radius

		# plotting the sphere on the image
		print("Showing the guess...")
		C = matplotlib.patches.Circle(center, radius=r, color='r', alpha=0.4)

		ax.add_patch(C)

		# The user can attempt to refine its guess
		tellme('Happy? Key click for yes, mouse click for no')

		if plt.waitforbuttonpress():
			plt.close()
			break

		C.remove()

	print(f"Result of interactive guess: {C}.\n --- END OF INTERACTIVE GUESS --- ")
	return circle(center[1], center[0], r, image=image)  # inverted coordinates
