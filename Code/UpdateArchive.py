from Code.archive import Archive
from Code.circle import circle
from PIL import Image

if __name__ == '__main__':

	A = Archive.load(Archive.PROMPT_DARIO)
	print(len(A))
	if input('Save changes? [y/n]') == 'y':
		A.save()
		print('Saved')
	else:
		print('Not saved')

	# Sample script for saving images
	# C: circle
	# im=None
	# for C in A:
	# 	if C.image_id == './Samples/prompt/prompt_10.png':
	# 		im = Image.fromarray(C.image, "RGB")
	# 		image_filename = './Samples/prompt/prompt_10.png'
	#
	#
	# im.show()
	# im.save(image_filename)
