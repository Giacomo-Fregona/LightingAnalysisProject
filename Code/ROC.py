import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from archive import Archive
from circle import circle
from figure_6 import getCouples

weight = {'l1m1' : 0.11, 'l10': 35, 'l11': 10, 'l2m2': 7, 'l2m1': 13, 'l20': 32, 'l21': 15, 'l22': 8}

def compute_roc(scores, labels, id):
    # compute ROC
    fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    # compute AUC
    roc_auc = auc(fpr, tpr)
    # plt.figure()
    lw = 2

    plt.plot(fpr, tpr,
             lw=lw, label=f'AUC = %0.2f' % roc_auc)
	# plt.plot(fpr, tpr, color='darkorange',
			 # lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    idx_tpr = np.where((fpr - 0.1) == min(i for i in (fpr - 0.1) if i > 0))
    print(f'{coeff}\t\tFor a FPR approximately equals to 0.1 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
    print('\t\tFor a FPR approximately equals to 0.1 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
    print('\t\tCheck FPR %0.2f' % fpr[idx_tpr[0][0]])


if __name__ == '__main__':
	scores = { coeff: [] for coeff in circle.coeffList + ['test_1']}
	labels = { coeff: [] for coeff in circle.coeffList + ['test_1']}


	for filename in [Archive.REAL, Archive.PROMPT, Archive.VARIATION]:


		print(filename)
		# Loading the archive
		A: Archive = Archive.load(filename)
		if filename == Archive.REAL:
			for i in [72, 62, 51, 47, 41, 39, 24, 20, 17, 14, 9]:
				A.pop(i)
		A_dict = A.asDict()

		# Retrieving all possible couples inside the same image
		couples = [couple for circlesList in A_dict.values() for couple in getCouples(circlesList)]
		print(len(couples))
		# Getting the data
		data1 = {coeff: [] for coeff in circle.coeffList}
		data2 = {coeff: [] for coeff in circle.coeffList}
		for (C1, C2) in couples:
			for coeff in circle.coeffList:
				data1[coeff].append(C1.get_coeff(coeff))
				data2[coeff].append(C2.get_coeff(coeff))

		diff = {coefff : None for coefff in circle.coeffList}
		for coeff in circle.coeffList[1:]:

			# Normalized version
			x = [RGB[d] / (data1['l00'][i][d]) for i, RGB in enumerate(data1[coeff]) for d in [0,1,2]]
			y = [RGB[d] / (data2['l00'][i][d]) for i, RGB in enumerate(data2[coeff]) for d in [0,1,2]]

			diff[coeff] = abs(np.array(x) - np.array(y))
		for coeff in circle.coeffList[1:]:
			for j in range(len(x)):
				score = diff[coeff][j]
				(scores[coeff]).append(score)

				label = A.fileName != Archive.REAL
				(labels[coeff]).append(label)
				print(f'Added {score} with label {label}')

		for j in range(len(x)):
			score = 0
			for i, coeff in enumerate(['l11', 'l22']):
				score += diff[coeff][j]


			label = A.fileName != Archive.REAL
			(labels['test_1']).append(label)
			print(f'Added {score} with label {label}')
			(scores['test_1']).append(score)

	for coeff in ['test_1']:
		compute_roc(scores[coeff], labels[coeff], coeff)

	plt.show()
