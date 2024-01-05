import matplotlib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from expectation_maximization import EM
from interaction_lib import interactiveGuess
from circle import circle
from archive import Archive
import copy


def correct_circle(A: Archive, archive_index: int = None, C: circle = None):
	"""
    Correcting a wrong circle in the archive
    @param C: Circle to be corrected
    @param A: The archive
    @param archive_index: The index of the circle in the archive (in images is represented inside square brackets)
    @return: None
    """

	if not (archive_index is None):
		oldC: circle = A[archive_index]
	else:
		oldC: circle = copy.deepcopy(C)

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
		A.pop(A.index(oldC))
		A.append(newC)
		A.save()
		print('Saved')
	else:
		print('Not saved')
		if input('Delete image? [y/n]') == 'y':
			A.pop(A.index(oldC))
			A.save()
			print('Deleted')
		else:
			print('Not deleted')


if __name__ == '__main__':
	A: Archive = Archive.load(Archive.VARIATION)
	A_dict = A.as_dict()
	print(A)
	A.file_name = Archive.VARIATION

	to_be_fixed_indexes = [49] # Indexes to be corrected
	to_be_deleted_indexes = [68, 71, 72, 73, 74, 75, 125, 126, 129, 130] # Indexes to be deleted
	to_be_deleted_images = [5, 1, 8, 13, 14]

	for image in to_be_deleted_images:
		for circle in A_dict[f'./Samples/variation/variation_{image}.png']:
			to_be_deleted_indexes.append(A.index(circle))

	to_be_fixed = [copy.deepcopy(A[index]) for index in to_be_fixed_indexes]
	to_be_deleted = [copy.deepcopy(A[index]) for index in to_be_deleted_indexes]
	for C in to_be_fixed:
		correct_circle(A, A.index(C))
		A = Archive.load(A.file_name)

	for C in to_be_deleted:
		A.pop(A.index(C))
		A.save()
		print('Deleted')
		A = Archive.load(A.file_name)

	print(A)







	# A.visualize()  # archive_index=18
# A.save()
