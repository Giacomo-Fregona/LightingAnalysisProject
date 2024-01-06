from matplotlib import collections as mc
import pickle
import matplotlib.pyplot as plt
import numpy as np
from archive import Archive

dataReal = Archive.load(Archive.REAL)
dataPrompt = Archive.load(Archive.PROMPT)

print("dataPrompt:")
print(dataPrompt)
print()

print("dataReal:")
print(dataReal)
print()


l00ListPrompt = [myC.l00 for myC in dataPrompt]
l1m1ListPrompt = [[myC.l1m1[k]/myC.l00[k] for k in range(3)] for myC in dataPrompt]
l10ListPrompt = [[myC.l10[k]/myC.l00[k] for k in range(3)] for myC in dataPrompt]
l11ListPrompt = [[myC.l11[k]/myC.l00[k] for k in range(3)] for myC in dataPrompt]
l2m2ListPrompt = [[myC.l2m2[k]/myC.l00[k] for k in range(3)] for myC in dataPrompt]
l2m1ListPrompt = [[myC.l2m1[k]/myC.l00[k] for k in range(3)] for myC in dataPrompt]
l20ListPrompt = [[myC.l20[k]/myC.l00[k] for k in range(3)] for myC in dataPrompt]
l21ListPrompt = [[myC.l21[k]/myC.l00[k] for k in range(3)] for myC in dataPrompt]
l22ListPrompt = [[myC.l22[k]/myC.l00[k] for k in range(3)] for myC in dataPrompt]

l00ListReal = [myC.l00 for myC in dataReal]
l1m1ListReal = [[myC.l1m1[k]/myC.l00[k] for k in range(3)] for myC in dataReal]
l10ListReal = [[myC.l10[k]/myC.l00[k] for k in range(3)] for myC in dataReal]
l11ListReal = [[myC.l11[k]/myC.l00[k] for k in range(3)] for myC in dataReal]
l2m2ListReal = [[myC.l2m2[k]/myC.l00[k] for k in range(3)] for myC in dataReal]
l2m1ListReal = [[myC.l2m1[k]/myC.l00[k] for k in range(3)] for myC in dataReal]
l20ListReal = [[myC.l20[k]/myC.l00[k] for k in range(3)] for myC in dataReal]
l21ListReal = [[myC.l21[k]/myC.l00[k] for k in range(3)] for myC in dataReal]
l22ListReal = [[myC.l22[k]/myC.l00[k] for k in range(3)] for myC in dataReal]


print(dataReal[0].image_id)

#Real
l1m1Q35Real = np.quantile(l1m1ListReal, 0.35, axis=0)
l1m1Q50Real = np.quantile(l1m1ListReal, 0.5, axis=0)
l1m1Q65Real = np.quantile(l1m1ListReal, 0.65, axis=0)
print(f"l1m1Q35Real: {l1m1Q35Real}")
print(f"l1m1Q50Real: {l1m1Q50Real}")
print(f"l1m1Q65Real: {l1m1Q65Real}")
print()
l10Q35Real = np.quantile(l10ListReal, 0.35, axis=0)
l10Q50Real = np.quantile(l10ListReal, 0.5, axis=0)
l10Q65Real = np.quantile(l10ListReal, 0.65, axis=0)
print(f"l10Q35Real: {l10Q35Real}")
print(f"l10Q50Real: {l10Q50Real}")
print(f"l10Q65Real: {l10Q65Real}")
print()
l11Q35Real = np.quantile(l11ListReal, 0.35, axis=0)
l11Q50Real = np.quantile(l11ListReal, 0.5, axis=0)
l11Q65Real = np.quantile(l11ListReal, 0.65, axis=0)
print(f"l11Q35Real: {l11Q35Real}")
print(f"l11Q50Real: {l11Q50Real}")
print(f"l11Q65Real: {l11Q65Real}")
print()
l2m2Q35Real = np.quantile(l2m2ListReal, 0.35, axis=0)
l2m2Q50Real = np.quantile(l2m2ListReal, 0.5, axis=0)
l2m2Q65Real = np.quantile(l2m2ListReal, 0.65, axis=0)
print(f"l2m2Q35Real: {l2m2Q35Real}")
print(f"l2m2Q50Real: {l2m2Q50Real}")
print(f"l2m2Q65Real: {l2m2Q65Real}")
print()
l2m1Q35Real = np.quantile(l2m1ListReal, 0.35, axis=0)
l2m1Q50Real = np.quantile(l2m1ListReal, 0.5, axis=0)
l2m1Q65Real = np.quantile(l2m1ListReal, 0.65, axis=0)
print(f"l2m1Q35Real: {l2m1Q35Real}")
print(f"l2m1Q50Real: {l2m1Q50Real}")
print(f"l2m1Q65Real: {l2m1Q65Real}")
print()
l20Q35Real = np.quantile(l20ListReal, 0.35, axis=0)
l20Q50Real = np.quantile(l20ListReal, 0.5, axis=0)
l20Q65Real = np.quantile(l20ListReal, 0.65, axis=0)
print(f"l20Q35Real: {l20Q35Real}")
print(f"l20Q50Real: {l20Q50Real}")
print(f"l20Q65Real: {l20Q65Real}")
print()
l21Q35Real = np.quantile(l21ListReal, 0.35, axis=0)
l21Q50Real = np.quantile(l21ListReal, 0.5, axis=0)
l21Q65Real = np.quantile(l21ListReal, 0.65, axis=0)
print(f"l21Q35Real: {l21Q35Real}")
print(f"l21Q50Real: {l21Q50Real}")
print(f"l21Q65Real: {l21Q65Real}")
print()

l22Q35Real = np.quantile(l22ListReal, 0.35, axis=0)
l22Q50Real = np.quantile(l22ListReal, 0.5, axis=0)
l22Q65Real = np.quantile(l22ListReal, 0.65, axis=0)
print(f"l22Q35Real: {l22Q35Real}")
print(f"l22Q50Real: {l22Q50Real}")
print(f"l22Q65Real: {l22Q65Real}")
print()

#Prompt
l1m1Q35Prompt = np.quantile(l1m1ListPrompt, 0.35, axis=0)
l1m1Q50Prompt = np.quantile(l1m1ListPrompt, 0.5, axis=0)
l1m1Q65Prompt = np.quantile(l1m1ListPrompt, 0.65, axis=0)
print(f"l1m1Q35Prompt: {l1m1Q35Prompt}")
print(f"l1m1Q50Prompt: {l1m1Q50Prompt}")
print(f"l1m1Q65Prompt: {l1m1Q65Prompt}")
print()
l10Q35Prompt = np.quantile(l10ListPrompt, 0.35, axis=0)
l10Q50Prompt = np.quantile(l10ListPrompt, 0.5, axis=0)
l10Q65Prompt = np.quantile(l10ListPrompt, 0.65, axis=0)
print(f"l10Q35Prompt: {l10Q35Prompt}")
print(f"l10Q50Prompt: {l10Q50Prompt}")
print(f"l10Q65Prompt: {l10Q65Prompt}")
print()
l11Q35Prompt = np.quantile(l11ListPrompt, 0.35, axis=0)
l11Q50Prompt = np.quantile(l11ListPrompt, 0.5, axis=0)
l11Q65Prompt = np.quantile(l11ListPrompt, 0.65, axis=0)
print(f"l11Q35Prompt: {l11Q35Prompt}")
print(f"l11Q50Prompt: {l11Q50Prompt}")
print(f"l11Q65Prompt: {l11Q65Prompt}")
print()
l2m2Q35Prompt = np.quantile(l2m2ListPrompt, 0.35, axis=0)
l2m2Q50Prompt = np.quantile(l2m2ListPrompt, 0.5, axis=0)
l2m2Q65Prompt = np.quantile(l2m2ListPrompt, 0.65, axis=0)
print(f"l2m2Q35Prompt: {l2m2Q35Prompt}")
print(f"l2m2Q50Prompt: {l2m2Q50Prompt}")
print(f"l2m2Q65Prompt: {l2m2Q65Prompt}")
print()
l2m1Q35Prompt = np.quantile(l2m1ListPrompt, 0.35, axis=0)
l2m1Q50Prompt = np.quantile(l2m1ListPrompt, 0.5, axis=0)
l2m1Q65Prompt = np.quantile(l2m1ListPrompt, 0.65, axis=0)
print(f"l2m1Q35Prompt: {l2m1Q35Prompt}")
print(f"l2m1Q50Prompt: {l2m1Q50Prompt}")
print(f"l2m1Q65Prompt: {l2m1Q65Prompt}")
print()
l20Q35Prompt = np.quantile(l20ListPrompt, 0.35, axis=0)
l20Q50Prompt = np.quantile(l20ListPrompt, 0.5, axis=0)
l20Q65Prompt = np.quantile(l20ListPrompt, 0.65, axis=0)
print(f"l20Q35Prompt: {l20Q35Prompt}")
print(f"l20Q50Prompt: {l20Q50Prompt}")
print(f"l20Q65Prompt: {l20Q65Prompt}")
print()
l21Q35Prompt = np.quantile(l21ListPrompt, 0.35, axis=0)
l21Q50Prompt = np.quantile(l21ListPrompt, 0.5, axis=0)
l21Q65Prompt = np.quantile(l21ListPrompt, 0.65, axis=0)
print(f"l21Q35Prompt: {l21Q35Prompt}")
print(f"l21Q50Prompt: {l21Q50Prompt}")
print(f"l21Q65Prompt: {l21Q65Prompt}")
print()

l22Q35Prompt = np.quantile(l22ListPrompt, 0.35, axis=0)
l22Q50Prompt = np.quantile(l22ListPrompt, 0.5, axis=0)
l22Q65Prompt = np.quantile(l22ListPrompt, 0.65, axis=0)
print(f"l22Q35Prompt: {l22Q35Prompt}")
print(f"l22Q50Prompt: {l22Q50Prompt}")
print(f"l22Q65Prompt: {l22Q65Prompt}")
print()


#c in RGBA (red, green, blue, alpha)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i in range(3):

	#Real and Prompt
	axes[i].plot(l1m1Q50Real[i], l1m1Q50Prompt[i], c=[0, 0, 0, 1], marker="o")
	axes[i].plot(l10Q50Real[i], l10Q50Prompt[i], c=[1, 0, 0, 1], marker="o")
	axes[i].plot(l11Q50Real[i], l11Q50Prompt[i], c=[0, 1, 0, 1], marker="o")
	axes[i].plot(l2m2Q50Real[i], l2m2Q50Prompt[i], c=[1, 1, 0, 1], marker="o")
	axes[i].plot(l2m1Q50Real[i], l2m1Q50Prompt[i], c=[0, 0, 1, 1], marker="o")
	axes[i].plot(l20Q50Real[i], l20Q50Prompt[i], c=[1, 0, 1, 1], marker="o")
	axes[i].plot(l21Q50Real[i], l21Q50Prompt[i], c=[0, 1, 1, 1], marker="o")
	axes[i].plot(l22Q50Real[i], l22Q50Prompt[i], c=[0, 0, 0, 0.5], marker="o")


	lines = []
	c = []
	# Real and Prompt
	lines.append([(l1m1Q35Real[i], l1m1Q50Prompt[i]), (l1m1Q65Real[i], l1m1Q50Prompt[i])])
	lines.append([(l1m1Q50Real[i], l1m1Q35Prompt[i]), (l1m1Q50Real[i], l1m1Q65Prompt[i])])
	c.append([0, 0, 0, 1])
	c.append([0, 0, 0, 1])
	lines.append([(l10Q35Real[i], l10Q50Prompt[i]), (l10Q65Real[i], l10Q50Prompt[i])])
	lines.append([(l10Q50Real[i], l10Q35Prompt[i]), (l10Q50Real[i], l10Q65Prompt[i])])
	c.append([1, 0, 0, 1])
	c.append([1, 0, 0, 1])
	lines.append([(l11Q35Real[i], l11Q50Prompt[i]), (l11Q65Real[i], l11Q50Prompt[i])])
	lines.append([(l11Q50Real[i], l11Q35Prompt[i]), (l11Q50Real[i], l11Q65Prompt[i])])
	c.append([0, 1, 0, 1])
	c.append([0, 1, 0, 1])
	lines.append([(l2m2Q35Real[i], l2m2Q50Prompt[i]), (l2m2Q65Real[i], l2m2Q50Prompt[i])])
	lines.append([(l2m2Q50Real[i], l2m2Q35Prompt[i]), (l2m2Q50Real[i], l2m2Q65Prompt[i])])
	c.append([1, 1, 0, 1])
	c.append([1, 1, 0, 1])
	lines.append([(l2m1Q35Real[i], l2m1Q50Prompt[i]), (l2m1Q65Real[i], l2m1Q50Prompt[i])])
	lines.append([(l2m1Q50Real[i], l2m1Q35Prompt[i]), (l2m1Q50Real[i], l2m1Q65Prompt[i])])
	c.append([0, 0, 1, 1])
	c.append([0, 0, 1, 1])
	lines.append([(l20Q35Real[i], l20Q50Prompt[i]), (l20Q65Real[i], l20Q50Prompt[i])])
	lines.append([(l20Q50Real[i], l20Q35Prompt[i]), (l20Q50Real[i], l20Q65Prompt[i])])
	c.append([1, 0, 1, 1])
	c.append([1, 0, 1, 1])
	lines.append([(l21Q35Real[i], l21Q50Prompt[i]), (l21Q65Real[i], l21Q50Prompt[i])])
	lines.append([(l21Q50Real[i], l21Q35Prompt[i]), (l21Q50Real[i], l21Q65Prompt[i])])
	c.append([0, 1, 1, 1])
	c.append([0, 1, 1, 1])
	lines.append([(l22Q35Real[i], l22Q50Prompt[i]), (l22Q65Real[i], l22Q50Prompt[i])])
	lines.append([(l22Q50Real[i], l22Q35Prompt[i]), (l22Q50Real[i], l22Q65Prompt[i])])
	c.append([0, 0, 0, 0.5])
	c.append([0, 0, 0, 0.5])



	lc = mc.LineCollection(lines, colors=c, linewidths=2)


	axes[i].add_collection(lc)
	axes[i].autoscale()
	axes[i].margins(0.05)

	lims = [
		np.min([axes[i].get_xlim(), axes[i].get_ylim()]),  # min of both axes[i]es
		np.max([axes[i].get_xlim(), axes[i].get_ylim()]),  # max of both axes[i]es
	]
	axes[i].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
	axes[i].set_aspect('equal')
	axes[i].set_xlim(lims)
	axes[i].set_ylim(lims)

plt.show()
