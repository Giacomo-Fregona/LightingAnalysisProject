import matplotlib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from expectation_maximization import EM
from interaction_lib import interactiveGuess
from circle import circle
from archive import Archive

def correct_circle(A: Archive, archive_index: int):
	"""
    Correcting a wrong circle in the archive
    @param A: The archive
    @param archive_index: The index of the circle in the archive (in images is represented inside square brackets)
    @return: None
    """

	oldC: circle = A[archive_index]

	# Representing the old circle
	matplotlib.rcParams['figure.figsize'] = [20, 7]
	plt.subplot(121)  # Subplot 1: only circle
	plt.title(f"Circle to be corrected")
	plt.imshow(oldC.onImage())
	plt.subplot(122)  # Subplot 2: rendered sphere
	plt.title("Actual rendering")
	plt.imshow(oldC.fastRenderedOnImage())
	plt.show()

	if input('Exit? [y/n]') == 'y':
		return

	# Obtaining a new circle
	originalImage = oldC.image.copy()
	newC = interactiveGuess(originalImage)
	if input('Only guess procedure? [y/n]') == 'y':
		pass
	else:
		newC = EM(originalImage, newC, rounds=10, visual=0, finalVisual=0, erase=1)
	newC.estimateCoefficients(originalImage, M=1000)
	newC.image_id = oldC.image_id
	newC.image = np.asarray(Image.open(newC.image_id), dtype=np.uint8)

	# Visualizing the result
	matplotlib.rcParams['figure.figsize'] = [20, 7]
	plt.subplot(121)  # Subplot 1: old rendering
	plt.title(f"Old rendering")
	plt.imshow(oldC.fastRenderedOnImage())
	plt.subplot(122)  # Subplot 2: new rendering
	plt.title("New rendering")
	plt.imshow(newC.fastRenderedOnImage())
	plt.show()

	# Eventually saving changes
	if input('Save changes? [y/n]') == 'y':
		A.pop(archive_index)
		A.append(newC)
		A.save()
		print('Saved')
	else:
		print('Not saved')
		if input('Delete image? [y/n]') == 'y':
			A.pop(archive_index)
			A.save()
			print('Deleted')
		else:
			print('Not deleted')


if __name__ == '__main__':


	A: Archive = Archive.load(Archive.PROMPT_DARIO)
	A.visualize() #archive_index=18
	# correct_circle(A, 18)

