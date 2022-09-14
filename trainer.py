import json
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words

class Trainer():
	def __init__(self):
		#constants
		self.IGNORE_WORDS = ["?", "!", ".", ","]
		self.BATCH_SIZE = 8
		self.HIDDEN_SIZE = 8
		self.LEARNING_RATE = 0.001
		self.NUM_EPOCHS = 1000
		self.NUM_WORKERS = 2
		self.FILE = "data.pth"

		self.input_size = 0
		self.output_size = 0
		#
		self.all_words = []
		self.tags = []
#		self.xy = []
		self.x_train, self.y_train = self.prepare_xy_train()

		self.model = ""

	def prepare_xy_train(self):
		xy = []
		x_train = []
		y_train = []

		with open("intents.json", "r") as f:
			intents = json.load(f)
		for intent in intents["intents"]:
			tag = intent['tag']
			self.tags.append(tag)
			for pattern in intent["patterns"]:
				w = tokenize(pattern)
				self.all_words.extend(w)
				xy.append((w, tag))
		self.all_words = [stem(w) for w in self.all_words if w not in self.IGNORE_WORDS]
		self.all_words = sorted(set(self.all_words))
		self.tags = sorted(set(self.tags))

		for (pattern_sentence, tag) in xy:
			bag = bag_of_words(pattern_sentence, self.all_words)
			x_train.append(bag)
			label = self.tags.index(tag)
			y_train.append(label)

		x_train = np.array(x_train)
		y_train = np.array(y_train)

		self.output_size = len(self.tags)
		self.input_size = len(x_train[0])

		return x_train, y_train

	def train_model(self):
		dataset = ChatDataSet(self.x_train, self.y_train)
		train_loader = DataLoader(dataset = dataset, batch_size = self.BATCH_SIZE, shuffle = True, num_workers = self.NUM_WORKERS)
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = NeuralNet(self.input_size, self.HIDDEN_SIZE, self.output_size).to(device)

		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr = self.LEARNING_RATE)

		for epoch in range(self.NUM_EPOCHS):
			for (words, labels) in train_loader:
				words = words.to(device)
				labels = labels.to(device)

				#forward
				outputs = self.model(words)
				loss = criterion(outputs, labels)

				#backwards and optimizer
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			if (epoch + 1) % 100 == 0:
				print(f'Epoch [{epoch + 1}], Loss: {loss.item():.4f}')

		print(f'Final loss: {loss.item():.4f}')

	def run(self):
		self.train_model()
		self.save_data()

	def save_data(self):
		data = {
			"model_state" : self.model.state_dict(),
			"input_size" : self.input_size,
			"output_size" : self.output_size,
			"hidden_size" : self.HIDDEN_SIZE,
			"all_words" : self.all_words,
			"tags" : self.tags
		}
		torch.save(data, self.FILE)
		print(f'Training complete. Data saved to {self.FILE}')



class ChatDataSet(Dataset):
	def __init__(self, x_train, y_train):
		self.n_samples = len(x_train)
		self.x_data = x_train
		self.y_data = y_train

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.n_samples

if __name__ == "__main__":
	trainer = Trainer()
	trainer.run()