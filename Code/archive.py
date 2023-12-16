"""
File containing methods for saving and retrieving data coming from the lighting analysis
"""

import pickle
from Code.circle import circle

class Archive(list):
	REAL = './Archive/real.pkl'
	PROMPT = './Archive/prompt.pkl'
	VARIATION = './Archive/variation.pkl'

	def __init__(self, file_name: str):
		self.file_name = file_name
		super().__init__()

	@staticmethod
	def load(which_archive):
		with open(which_archive, 'rb+') as file:
			return pickle.load(file)

	def save(self):
		with open(self.file_name, 'wb') as file:
			return pickle.dump(self, file)


if __name__ == '__main__':
	# Archive bootstrapping
	real_archive = Archive(Archive.REAL)
	real_archive.save()

	prompt_archive = Archive(Archive.PROMPT)
	prompt_archive.save()

	variation_archive = Archive(Archive.VARIATION)
	variation_archive.save()
