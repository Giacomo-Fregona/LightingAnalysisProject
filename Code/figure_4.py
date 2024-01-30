from matplotlib import collections as mc
import pickle
import matplotlib.pyplot as plt
import numpy as np
from archive import Archive
import sys
sys.path.append('.')

def rpv(): # r=real, p=prompt, v=variation
	flag1 = ""
	while True:
		flag1 = input("real, prompt or variation? ").lower().strip()
		if flag1 == "real" or flag1 == "prompt" or flag1 == "variation":
			break
		else:
			print("Error: Answer must be real, prompt or variation")
	flag2 = ""
	
	flag2_prompt = {"real": "prompt or variation", "prompt": "real or variation", "variation": "real or prompt"}[flag1]
	while True:
		flag2 = input(f"{flag2_prompt}? ").lower().strip()
		if (flag2 == "real" or flag2 == "prompt" or flag2 == "variation") and (flag1 != flag2):
			break
		else:
			print(f"Error: Answer must be {flag2_prompt}")
	return [flag1, flag2]

[flag1, flag2] = rpv()

dataX = Archive.load({"real": Archive.REAL, "prompt": Archive.PROMPT, "variation": Archive.VARIATION}[flag1])
dataY = Archive.load({"real": Archive.REAL, "prompt": Archive.PROMPT, "variation": Archive.VARIATION}[flag2])

print(f"{flag1}:")
print(dataX) #print the Archive
print()

print(f"{flag2}:")
print(dataY) #print the Archive
print()


l00ListX = [myC.l00 for myC in dataX]
l1m1ListX = [[myC.l1m1[k]/myC.l00[k] for k in range(3)] for myC in dataX]
l10ListX = [[myC.l10[k]/myC.l00[k] for k in range(3)] for myC in dataX]
l11ListX = [[myC.l11[k]/myC.l00[k] for k in range(3)] for myC in dataX]
l2m2ListX = [[myC.l2m2[k]/myC.l00[k] for k in range(3)] for myC in dataX]
l2m1ListX = [[myC.l2m1[k]/myC.l00[k] for k in range(3)] for myC in dataX]
l20ListX = [[myC.l20[k]/myC.l00[k] for k in range(3)] for myC in dataX]
l21ListX = [[myC.l21[k]/myC.l00[k] for k in range(3)] for myC in dataX]
l22ListX = [[myC.l22[k]/myC.l00[k] for k in range(3)] for myC in dataX]

l00ListY = [myC.l00 for myC in dataY]
l1m1ListY = [[myC.l1m1[k]/myC.l00[k] for k in range(3)] for myC in dataY]
l10ListY = [[myC.l10[k]/myC.l00[k] for k in range(3)] for myC in dataY]
l11ListY = [[myC.l11[k]/myC.l00[k] for k in range(3)] for myC in dataY]
l2m2ListY = [[myC.l2m2[k]/myC.l00[k] for k in range(3)] for myC in dataY]
l2m1ListY = [[myC.l2m1[k]/myC.l00[k] for k in range(3)] for myC in dataY]
l20ListY = [[myC.l20[k]/myC.l00[k] for k in range(3)] for myC in dataY]
l21ListY = [[myC.l21[k]/myC.l00[k] for k in range(3)] for myC in dataY]
l22ListY = [[myC.l22[k]/myC.l00[k] for k in range(3)] for myC in dataY]

#X
l1m1Q35X = np.quantile(l1m1ListX, 0.35, axis=0)
l1m1Q50X = np.quantile(l1m1ListX, 0.5, axis=0)
l1m1Q65X = np.quantile(l1m1ListX, 0.65, axis=0)
print(f"l1m1Q35X: {l1m1Q35X}")
print(f"l1m1Q50X: {l1m1Q50X}")
print(f"l1m1Q65X: {l1m1Q65X}")
print()
l10Q35X = np.quantile(l10ListX, 0.35, axis=0)
l10Q50X = np.quantile(l10ListX, 0.5, axis=0)
l10Q65X = np.quantile(l10ListX, 0.65, axis=0)
print(f"l10Q35X: {l10Q35X}")
print(f"l10Q50X: {l10Q50X}")
print(f"l10Q65X: {l10Q65X}")
print()
l11Q35X = np.quantile(l11ListX, 0.35, axis=0)
l11Q50X = np.quantile(l11ListX, 0.5, axis=0)
l11Q65X = np.quantile(l11ListX, 0.65, axis=0)
print(f"l11Q35X: {l11Q35X}")
print(f"l11Q50X: {l11Q50X}")
print(f"l11Q65X: {l11Q65X}")
print()
l2m2Q35X = np.quantile(l2m2ListX, 0.35, axis=0)
l2m2Q50X = np.quantile(l2m2ListX, 0.5, axis=0)
l2m2Q65X = np.quantile(l2m2ListX, 0.65, axis=0)
print(f"l2m2Q35X: {l2m2Q35X}")
print(f"l2m2Q50X: {l2m2Q50X}")
print(f"l2m2Q65X: {l2m2Q65X}")
print()
l2m1Q35X = np.quantile(l2m1ListX, 0.35, axis=0)
l2m1Q50X = np.quantile(l2m1ListX, 0.5, axis=0)
l2m1Q65X = np.quantile(l2m1ListX, 0.65, axis=0)
print(f"l2m1Q35X: {l2m1Q35X}")
print(f"l2m1Q50X: {l2m1Q50X}")
print(f"l2m1Q65X: {l2m1Q65X}")
print()
l20Q35X = np.quantile(l20ListX, 0.35, axis=0)
l20Q50X = np.quantile(l20ListX, 0.5, axis=0)
l20Q65X = np.quantile(l20ListX, 0.65, axis=0)
print(f"l20Q35X: {l20Q35X}")
print(f"l20Q50X: {l20Q50X}")
print(f"l20Q65X: {l20Q65X}")
print()
l21Q35X = np.quantile(l21ListX, 0.35, axis=0)
l21Q50X = np.quantile(l21ListX, 0.5, axis=0)
l21Q65X = np.quantile(l21ListX, 0.65, axis=0)
print(f"l21Q35X: {l21Q35X}")
print(f"l21Q50X: {l21Q50X}")
print(f"l21Q65X: {l21Q65X}")
print()

l22Q35X = np.quantile(l22ListX, 0.35, axis=0)
l22Q50X = np.quantile(l22ListX, 0.5, axis=0)
l22Q65X = np.quantile(l22ListX, 0.65, axis=0)
print(f"l22Q35X: {l22Q35X}")
print(f"l22Q50X: {l22Q50X}")
print(f"l22Q65X: {l22Q65X}")
print()

#Y
l1m1Q35Y = np.quantile(l1m1ListY, 0.35, axis=0)
l1m1Q50Y = np.quantile(l1m1ListY, 0.5, axis=0)
l1m1Q65Y = np.quantile(l1m1ListY, 0.65, axis=0)
print(f"l1m1Q35Y: {l1m1Q35Y}")
print(f"l1m1Q50Y: {l1m1Q50Y}")
print(f"l1m1Q65Y: {l1m1Q65Y}")
print()
l10Q35Y = np.quantile(l10ListY, 0.35, axis=0)
l10Q50Y = np.quantile(l10ListY, 0.5, axis=0)
l10Q65Y = np.quantile(l10ListY, 0.65, axis=0)
print(f"l10Q35Y: {l10Q35Y}")
print(f"l10Q50Y: {l10Q50Y}")
print(f"l10Q65Y: {l10Q65Y}")
print()
l11Q35Y = np.quantile(l11ListY, 0.35, axis=0)
l11Q50Y = np.quantile(l11ListY, 0.5, axis=0)
l11Q65Y = np.quantile(l11ListY, 0.65, axis=0)
print(f"l11Q35Y: {l11Q35Y}")
print(f"l11Q50Y: {l11Q50Y}")
print(f"l11Q65Y: {l11Q65Y}")
print()
l2m2Q35Y = np.quantile(l2m2ListY, 0.35, axis=0)
l2m2Q50Y = np.quantile(l2m2ListY, 0.5, axis=0)
l2m2Q65Y = np.quantile(l2m2ListY, 0.65, axis=0)
print(f"l2m2Q35Y: {l2m2Q35Y}")
print(f"l2m2Q50Y: {l2m2Q50Y}")
print(f"l2m2Q65Y: {l2m2Q65Y}")
print()
l2m1Q35Y = np.quantile(l2m1ListY, 0.35, axis=0)
l2m1Q50Y = np.quantile(l2m1ListY, 0.5, axis=0)
l2m1Q65Y = np.quantile(l2m1ListY, 0.65, axis=0)
print(f"l2m1Q35Y: {l2m1Q35Y}")
print(f"l2m1Q50Y: {l2m1Q50Y}")
print(f"l2m1Q65Y: {l2m1Q65Y}")
print()
l20Q35Y = np.quantile(l20ListY, 0.35, axis=0)
l20Q50Y = np.quantile(l20ListY, 0.5, axis=0)
l20Q65Y = np.quantile(l20ListY, 0.65, axis=0)
print(f"l20Q35Y: {l20Q35Y}")
print(f"l20Q50Y: {l20Q50Y}")
print(f"l20Q65Y: {l20Q65Y}")
print()
l21Q35Y = np.quantile(l21ListY, 0.35, axis=0)
l21Q50Y = np.quantile(l21ListY, 0.5, axis=0)
l21Q65Y = np.quantile(l21ListY, 0.65, axis=0)
print(f"l21Q35Y: {l21Q35Y}")
print(f"l21Q50Y: {l21Q50Y}")
print(f"l21Q65Y: {l21Q65Y}")
print()

l22Q35Y = np.quantile(l22ListY, 0.35, axis=0)
l22Q50Y = np.quantile(l22ListY, 0.5, axis=0)
l22Q65Y = np.quantile(l22ListY, 0.65, axis=0)
print(f"l22Q35Y: {l22Q35Y}")
print(f"l22Q50Y: {l22Q50Y}")
print(f"l22Q65Y: {l22Q65Y}")
print()


print(f"l1m1X: {max([np.abs(l1m1Q50X[0]-l1m1Q50X[1]), np.abs(l1m1Q50X[0]-l1m1Q50X[2]), np.abs(l1m1Q50X[1]-l1m1Q50X[2])])}")
print(f"l10X: {max([np.abs(l10Q50X[0]-l10Q50X[1]), np.abs(l10Q50X[0]-l10Q50X[2]), np.abs(l10Q50X[1]-l10Q50X[2])])}")
print(f"l11X: {max([np.abs(l11Q50X[0]-l11Q50X[1]), np.abs(l11Q50X[0]-l11Q50X[2]), np.abs(l11Q50X[1]-l11Q50X[2])])}")
print(f"l2m2X: {max([np.abs(l2m2Q50X[0]-l2m2Q50X[1]), np.abs(l2m2Q50X[0]-l2m2Q50X[2]), np.abs(l2m2Q50X[1]-l2m2Q50X[2])])}")
print(f"l2m1X: {max([np.abs(l2m1Q50X[0]-l2m1Q50X[1]), np.abs(l2m1Q50X[0]-l2m1Q50X[2]), np.abs(l2m1Q50X[1]-l2m1Q50X[2])])}")
print(f"l20X: {max([np.abs(l20Q50X[0]-l20Q50X[1]), np.abs(l20Q50X[0]-l20Q50X[2]), np.abs(l20Q50X[1]-l20Q50X[2])])}")
print(f"l21X: {max([np.abs(l21Q50X[0]-l21Q50X[1]), np.abs(l21Q50X[0]-l21Q50X[2]), np.abs(l21Q50X[1]-l21Q50X[2])])}")
print(f"l22X: {max([np.abs(l22Q50X[0]-l22Q50X[1]), np.abs(l22Q50X[0]-l22Q50X[2]), np.abs(l22Q50X[1]-l22Q50X[2])])}")
print()
print(f"l1m1Y: {max([np.abs(l1m1Q50Y[0]-l1m1Q50Y[1]), np.abs(l1m1Q50Y[0]-l1m1Q50Y[2]), np.abs(l1m1Q50Y[1]-l1m1Q50Y[2])])}")
print(f"l10Y: {max([np.abs(l10Q50Y[0]-l10Q50Y[1]), np.abs(l10Q50Y[0]-l10Q50Y[2]), np.abs(l10Q50Y[1]-l10Q50Y[2])])}")
print(f"l11Y: {max([np.abs(l11Q50Y[0]-l11Q50Y[1]), np.abs(l11Q50Y[0]-l11Q50Y[2]), np.abs(l11Q50Y[1]-l11Q50Y[2])])}")
print(f"l2m2Y: {max([np.abs(l2m2Q50Y[0]-l2m2Q50Y[1]), np.abs(l2m2Q50Y[0]-l2m2Q50Y[2]), np.abs(l2m2Q50Y[1]-l2m2Q50Y[2])])}")
print(f"l2m1Y: {max([np.abs(l2m1Q50Y[0]-l2m1Q50Y[1]), np.abs(l2m1Q50Y[0]-l2m1Q50Y[2]), np.abs(l2m1Q50Y[1]-l2m1Q50Y[2])])}")
print(f"l20Y: {max([np.abs(l20Q50Y[0]-l20Q50Y[1]), np.abs(l20Q50Y[0]-l20Q50Y[2]), np.abs(l20Q50Y[1]-l20Q50Y[2])])}")
print(f"l21Y: {max([np.abs(l21Q50Y[0]-l21Q50Y[1]), np.abs(l21Q50Y[0]-l21Q50Y[2]), np.abs(l21Q50Y[1]-l21Q50Y[2])])}")
print(f"l22Y: {max([np.abs(l22Q50Y[0]-l22Q50Y[1]), np.abs(l22Q50Y[0]-l22Q50Y[2]), np.abs(l22Q50Y[1]-l22Q50Y[2])])}")
print()

differencesX = [0]*3
differencesY = [0]*3
for i in range(3):
	differencesX[i] = np.mean([	l1m1Q65X[i]-l1m1Q35X[i],
				l10Q65X[i]-l10Q35X[i],
				l11Q65X[i]-l11Q35X[i],
				l2m2Q65X[i]-l2m2Q35X[i],
				l2m1Q65X[i]-l2m1Q35X[i],
				l20Q65X[i]-l20Q35X[i],
				l21Q65X[i]-l21Q35X[i],
				l22Q65X[i]-l22Q35X[i]])
	differencesY[i] = np.mean([	l1m1Q65Y[i]-l1m1Q35Y[i],
				l10Q65Y[i]-l10Q35Y[i],
				l11Q65Y[i]-l11Q35Y[i],
				l2m2Q65Y[i]-l2m2Q35Y[i],
				l2m1Q65Y[i]-l2m1Q35Y[i],
				l20Q65Y[i]-l20Q35Y[i],
				l21Q65Y[i]-l21Q35Y[i],
				l22Q65Y[i]-l22Q35Y[i]])
	
print(f"mean of differences 65 and 35 quantile X = {differencesX}")
print(f"mean of differences 65 and 35 quantile Y = {differencesY}")
print()

#c in RGBA (red, green, blue, alpha)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i in range(3):
	
	#Y and X
	axes[i].plot(l1m1Q50X[i], l1m1Q50Y[i], c=[0, 0, 1, 1], marker="o", label="l(1,-1)")
	axes[i].plot(l10Q50X[i], l10Q50Y[i], c=[1, 0, 0, 1], marker="o", label="l(1,0)")
	axes[i].plot(l11Q50X[i], l11Q50Y[i], c=[1, 0.9, 0.1, 1], marker="o", label="l(1,1)")
	axes[i].plot(l2m2Q50X[i], l2m2Q50Y[i], c=[1, 0, 1, 1], marker="o", label="l(2,-2)")
	axes[i].plot(l2m1Q50X[i], l2m1Q50Y[i], c=[0, 1, 0, 1], marker="o", label="l(2,-1)")
	axes[i].plot(l20Q50X[i], l20Q50Y[i], c=[0, 1, 1, 1], marker="o", label="l(2,0)")
	axes[i].plot(l21Q50X[i], l21Q50Y[i], c=[0.7, 0, 0, 1], marker="o", label="l(2,1)")
	axes[i].plot(l22Q50X[i], l22Q50Y[i], c=[0.5, 0.5, 0.5, 1], marker="o", label="l(2,2)")


	lines = []
	c = []
	# Y and X
	lines.append([(l1m1Q35X[i], l1m1Q50Y[i]), (l1m1Q65X[i], l1m1Q50Y[i])])
	lines.append([(l1m1Q50X[i], l1m1Q35Y[i]), (l1m1Q50X[i], l1m1Q65Y[i])])
	c.append([0, 0, 1, 1])
	c.append([0, 0, 1, 1])
	lines.append([(l10Q35X[i], l10Q50Y[i]), (l10Q65X[i], l10Q50Y[i])])
	lines.append([(l10Q50X[i], l10Q35Y[i]), (l10Q50X[i], l10Q65Y[i])])
	c.append([1, 0, 0, 1])
	c.append([1, 0, 0, 1])
	lines.append([(l11Q35X[i], l11Q50Y[i]), (l11Q65X[i], l11Q50Y[i])])
	lines.append([(l11Q50X[i], l11Q35Y[i]), (l11Q50X[i], l11Q65Y[i])])
	c.append([1, 0.9, 0.1, 1])
	c.append([1, 0.9, 0.1, 1])
	lines.append([(l2m2Q35X[i], l2m2Q50Y[i]), (l2m2Q65X[i], l2m2Q50Y[i])])
	lines.append([(l2m2Q50X[i], l2m2Q35Y[i]), (l2m2Q50X[i], l2m2Q65Y[i])])
	c.append([1, 0, 1, 1])
	c.append([1, 0, 1, 1])
	lines.append([(l2m1Q35X[i], l2m1Q50Y[i]), (l2m1Q65X[i], l2m1Q50Y[i])])
	lines.append([(l2m1Q50X[i], l2m1Q35Y[i]), (l2m1Q50X[i], l2m1Q65Y[i])])
	c.append([0, 1, 0, 1])
	c.append([0, 1, 0, 1])
	lines.append([(l20Q35X[i], l20Q50Y[i]), (l20Q65X[i], l20Q50Y[i])])
	lines.append([(l20Q50X[i], l20Q35Y[i]), (l20Q50X[i], l20Q65Y[i])])
	c.append([0, 1, 1, 1])
	c.append([0, 1, 1, 1])
	lines.append([(l21Q35X[i], l21Q50Y[i]), (l21Q65X[i], l21Q50Y[i])])
	lines.append([(l21Q50X[i], l21Q35Y[i]), (l21Q50X[i], l21Q65Y[i])])
	c.append([0.7, 0, 0, 1])
	c.append([0.7, 0, 0, 1])
	lines.append([(l22Q35X[i], l22Q50Y[i]), (l22Q65X[i], l22Q50Y[i])])
	lines.append([(l22Q50X[i], l22Q35Y[i]), (l22Q50X[i], l22Q65Y[i])])
	c.append([0.5, 0.5, 0.5, 1])
	c.append([0.5, 0.5, 0.5, 1])



	lc = mc.LineCollection(lines, colors=c, linewidths=1.5)

	axes[i].add_collection(lc)
	axes[i].autoscale()
	axes[i].margins(0.05)
	
	lims = [
		np.min([l1m1Q35X, l10Q35X, l11Q35X, l2m2Q35X, l2m1Q35X, l20Q35X, l21Q35X, l22Q35X, l1m1Q35Y, l10Q35Y, l11Q35Y, l2m2Q35Y, l2m1Q35Y, l20Q35Y, l21Q35Y, l22Q35Y])-0.1,  # min of both axes[i]es
		np.max([l1m1Q65X, l10Q65X, l11Q65X, l2m2Q65X, l2m1Q65X, l20Q65X, l21Q65X, l22Q65X, l1m1Q65Y, l10Q65Y, l11Q65Y, l2m2Q65Y, l2m1Q65Y, l20Q65Y, l21Q65Y, l22Q65Y])+0.1,  # max of both axes[i]es
	]
	
	axes[i].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
	axes[i].set_aspect('equal')
	axes[i].set_xlim(lims)
	axes[i].set_ylim(lims)
	axes[i].legend(loc="upper left")
	if i==0:
		axes[i].set_xlabel(f"{flag1}")
		axes[i].set_ylabel(f"{flag2}")
	
	if i==0:
		axes[i].set_title(label="Red layer")
	elif i==1:
		axes[i].set_title("Green layer")
	elif i==2:
		axes[i].set_title("Blue layer")
	
plt.show()
