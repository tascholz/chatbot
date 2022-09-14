import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

class Chatbot():
	def __init__(self):
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		with open('intents.json', 'r') as f:
			self.intents = json.load(f)
		self.data = torch.load("data.pth")
		
		self.input_size = self.data["input_size"]
		self.hidden_size = self.data["hidden_size"]
		self.output_size = self.data["output_size"]
		self.all_words = self.data["all_words"]
		self.tags = self.data["tags"]
		self.model_state = self.data["model_state"]

		self.model = NeuralNet(self.input_size, self.hidden_size, self.output_size).to(self.device) 
		self.model.load_state_dict(self.model_state)
		self.model.eval()

		self.bot_name = "Bot"

	def run(self):
		print(f'{self.bot_name}: How can I help you?(quit to exit)')
		print("")

		while True:
			sentence = input("You: ")
			if sentence == "quit":
				break
			tag, prediction = self.process_input(sentence)
			output = self.generate_output(tag, prediction)
			self.show_output(output)

	def process_input(self, sentence):
		sentence = tokenize(sentence)
		x = bag_of_words(sentence, self.all_words)
		x = x.reshape(1, x.shape[0])
		x = torch.from_numpy(x)

		output = self.model(x)
		_, predicted = torch.max(output, dim = 1)
		tag = self.tags[predicted.item()]
		probs = torch.softmax(output, dim = 1)
		prediction = probs[0][predicted.item()]
		return tag, prediction

	def generate_output(self, tag, prediction):
		if prediction.item() > 0.75:
			for intent in self.intents["intents"]:
				if tag == intent["tag"]:
					return f'{self.bot_name}: {random.choice(intent["responses"])}'
		else:
			return f'{self.bot_name}: I do not understand. Please call xxx-xxxxxx for human support'

	def show_output(self, sentence):
		print(sentence)
		print("")

if __name__ == "__main__":
	bot = Chatbot()
	bot.run()