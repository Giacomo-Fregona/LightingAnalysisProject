import pickle

flag = ""
while True:
	flag = input("real, prompt or variation? ").lower().strip()
	if flag == "real" or flag == "prompt" or flag == "variation":
		break
	else:
		print("Error: Answer must be real or dalle2")
		
fileToOpen = "./Archive/"+flag+".pkl"
with open(fileToOpen, "rb") as file:
	data = pickle.load(file)
file.close()

print('Showing the pickled data:')

print(data)
