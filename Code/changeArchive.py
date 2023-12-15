import pickle
import os

from archive import Archive

for flag in ["real", "prompt", "variation"]:
	fileToOpen = "./Archive/"+flag+".pkl"
	
	with open(fileToOpen, "rb") as file:
		data = pickle.load(file)
	file.close()
	
	print(data)
	
	dataToUpload=[]
	
	for element in data:
		for key, value in element.items():
			dataToUpload.append(value)
	
	if (len(data) == len(dataToUpload)): # This check is a trivial way to ensure the correctness of the algorithms just completed
		os.remove(fileToOpen)
		if (flag == "real"):
			real_archive = Archive(Archive.REAL)
			real_archive.save()
		elif (flag == "prompt"):
			prompt_archive = Archive(Archive.PROMPT)
			prompt_archive.save()
		elif (flag == "variation"):
			variation_archive = Archive(Archive.VARIATION)
			variation_archive.save()
		if (flag == "real"):
			pa: Archive = Archive.load(Archive.REAL)
		elif (flag == "prompt"):
			pa: Archive = Archive.load(Archive.PROMPT)
		elif (flag == "variation"):
			pa: Archive = Archive.load(Archive.VARIATION)
		for i in range(len(dataToUpload)):
			pa.append(dataToUpload[i])
			pa.save()
			print(f"Caricati {i}/{len(dataToUpload)}")
	
