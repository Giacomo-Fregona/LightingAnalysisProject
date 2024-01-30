from PIL import Image

import circle
import pickle
import numpy as np
from interaction_lib import interactiveGuess
# Borderline cases are not managed (e.g. case with np.arctan(1/0)). Indeed, they are extremely rare and would make the code much more difficult to read

def positionSphereFromCamera(C, FOVX=90, FOVY=90, pixelLength = 1024): #FOV=field of view
	phi = ((180  - FOVX) / 2) + ((C.center.x * FOVX) / pixelLength)
	theta = ((180 - FOVY) / 2) + ((C.center.y * FOVY) / pixelLength)
	return [phi, theta]

def positionLightfromSphereRGB(C):
	R = [C.l1m1[0]/C.l00[0], C.l10[0]/C.l00[0], C.l11[0]/C.l00[0]] #[l1m1, l10, l11] normalized red
	G = [C.l1m1[1]/C.l00[1], C.l10[1]/C.l00[1], C.l11[1]/C.l00[1]] #[l1m1, l10, l11] normalized green
	B = [C.l1m1[2]/C.l00[2], C.l10[2]/C.l00[2], C.l11[2]/C.l00[2]] #[l1m1, l10, l11] normalized blue
	angleR = [np.rad2deg(np.arctan(R[1]/R[0])), np.rad2deg(np.arctan(np.sqrt(R[0]**2+R[1]**2)/R[2]))]
	angleG = [np.rad2deg(np.arctan(G[1]/G[0])), np.rad2deg(np.arctan(np.sqrt(G[0]**2+G[1]**2)/G[2]))]
	angleB = [np.rad2deg(np.arctan(B[1]/B[0])), np.rad2deg(np.arctan(np.sqrt(B[0]**2+B[1]**2)/B[2]))]
	return [angleR, angleG, angleB] #3x3

def estimationPositionLightFromSphereFromCamera(SphereFromCamera, LightfromSphere):
	return [270 + SphereFromCamera[0] - LightfromSphere[0], 180 + SphereFromCamera[1] - LightfromSphere[1]]


# if __name__ == '__main__':
	# flag = ""
	# while True:
	# 	flag = input("real, prompt or variation? ").lower().strip()
	# 	if flag == "real" or flag == "prompt" or flag == "variation":
	# 		break
	# 	else:
	# 		print("Error: Answer must be real or dalle2")
	#
	# fileToOpen = "./Archive/"+flag+".pkl"
	# with open(fileToOpen, "rb") as file:
	# 	data = pickle.load(file)
	# file.close()
	
	
	# listOfCircleInSameImage = [data[0]]
	# for i in range(1, len(data)):
	# 	if data[i].image_id == listOfCircleInSameImage[0].image_id:
	# 		listOfCircleInSameImage.append(data[i])
	# 	else:
	# 		if len(listOfCircleInSameImage) == 1:
	# 			print(listOfCircleInSameImage[0].image_id)
	# 			print("NON CONTARE: UNA SOLA SFERA NELL'IMMAGINE")
	# 			print()
	# 		else:
	# 			print(f"Image: {listOfCircleInSameImage[0].image_id}")
	# 			GeneralR = []
	# 			GeneralG = []
	# 			GeneralB = []
	# 			for j in range(len(listOfCircleInSameImage)):
	# 				print(f"Circle: {listOfCircleInSameImage[j].center}, {listOfCircleInSameImage[j].r}")
	# 				sfc = positionSphereFromCamera(listOfCircleInSameImage[j])
	# 				print(f"Sphere from Camera: {sfc}")
	# 				lfs = positionLightfromSphereRGB(listOfCircleInSameImage[j])
	# 				print(f"Light from SphereR: {lfs[0]}")
	# 				print(f"Light from SphereG: {lfs[1]}")
	# 				print(f"Light from SphereB: {lfs[2]}")
	# 				lfcR = estimationPositionLightFromSphereFromCamera(sfc, lfs[0])
	# 				GeneralR.append(lfcR)
	# 				lfcG = estimationPositionLightFromSphereFromCamera(sfc, lfs[1])
	# 				GeneralG.append(lfcG)
	# 				lfcB = estimationPositionLightFromSphereFromCamera(sfc, lfs[2])
	# 				GeneralB.append(lfcB)
	# 				print(f"Light from Camera R: {lfcR}")
	# 				print(f"Light from Camera G: {lfcG}")
	# 				print(f"Light from Camera B: {lfcB}")
	# 				print()
	# 			print(f"Rvar : {np.std(GeneralR, axis=0)}")
	# 			print(f"Gvar : {np.std(GeneralG, axis=0)}")
	# 			print(f"Bvar : {np.std(GeneralB, axis=0)}")
	# 			print()
	#
	# 		listOfCircleInSameImage = [data[i]]
if __name__ == '__main__':
	originalImage = np.asarray(Image.open('./Samples/real/test.jpg'), dtype=np.uint8)


	C1: circle
	C1 = interactiveGuess(originalImage)
	C1.estimateCoefficients(image=originalImage, M=-1)
	C2: circle
	C2 = interactiveGuess(originalImage)
	C2.estimateCoefficients(image=originalImage, M=-1)

	sfc1 = positionSphereFromCamera(C1, pixelLength=1920)
	sfc2 = positionSphereFromCamera(C2, pixelLength=1920)

	lfs1 = positionLightfromSphereRGB(C1)
	lfs2 = positionLightfromSphereRGB(C2)

	lfcR1 = estimationPositionLightFromSphereFromCamera(sfc1, lfs1[0])
	lfcR2 = estimationPositionLightFromSphereFromCamera(sfc2, lfs1[0])

	lfcG1 = estimationPositionLightFromSphereFromCamera(sfc1, lfs1[1])
	lfcG2 = estimationPositionLightFromSphereFromCamera(sfc2, lfs2[1])

	lfcB1 = estimationPositionLightFromSphereFromCamera(sfc1, lfs1[2])
	lfcB2 = estimationPositionLightFromSphereFromCamera(sfc2, lfs2[2])

	print(f'{lfcR1} |||| {lfcR2} ---- {lfs1[0]} |||| {lfs2[0]}')
	print(f'{lfcG1} |||| {lfcG2} ---- {lfs1[1]} |||| {lfs2[1]}')
	print(f'{lfcB1} |||| {lfcB2} ---- {lfs1[2]} |||| {lfs2[2]}')





















