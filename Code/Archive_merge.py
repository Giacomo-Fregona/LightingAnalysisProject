import matplotlib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from expectation_maximization import EM
from interaction_lib import interactiveGuess
from circle import circle
from archive import Archive
import copy

if __name__ == '__main__':
	A: Archive = Archive.load(Archive.PROMPT)
	D: Archive = Archive.load(Archive.PROMPT_DARIO)
	G: Archive = Archive.load(Archive.PROMPT_GIACOMO)
	print(A, D, G)

	A_images = set()
	D_images: set = {0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12}
	G_images: set = {17, 16, 15, 24, 13, 25, 14, 31, 36, 33, 30, 35, 38, 26, 29, 37, 32, 18, 20, 21, 22, 23, 27, 34}

	# Checking the sets are disjoint
	if len((A_images.union(D_images)).union(G_images)) != (len(A_images) + len(D_images) + len(G_images)):
		raise Exception('Nontrivial intersection')

	A_images = set(f'./Samples/prompt/prompt_{number}.png' for number in A_images)
	D_images = set(f'./Samples/prompt/prompt_{number}.png' for number in D_images)
	G_images = set(f'./Samples/prompt/prompt_{number}.png' for number in G_images)

	# Removing from archive the undesired images
	to_remove = []
	C: circle
	for C in A:
		if C.image_id in A_images:
			pass
		else:
			to_remove.append(C)
	to_remove.reverse()
	for C in to_remove:
		print(f'Removing circle with id {C.image_id}')
		A.pop(A.index(C))
	print(A)

	# Adding from the other archives
	D_dict = D.as_dict()
	for image_id in D_images:
		for C in D_dict[image_id]:
			A.append(copy.deepcopy(C))
			print(f'Added')

	print('Added from D:', A)

	G_dict = G.as_dict()
	for image_id in G_images:
		for C in G_dict[image_id]:
			A.append(copy.deepcopy(C))
			print(f'Added')

	print('Added from G:', A)

	if input('Save changes? [y/n]') == 'y':
		A.save()
		print('Saved')
	else:
		print('Not saved')
