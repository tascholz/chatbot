import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from telegram.ext import Updater, CallbackContext, CommandHandler, MessageHandler, Filters
from telegram import Update

class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

	def runConsole(self):
		print(Bcolors.OKGREEN)
		print(f'{self.bot_name}: How can I help you?(quit to exit)')
		print("")
		print(Bcolors.ENDC)
		while True:
			print(Bcolors.WARNING)
			sentence = input("You: ")
			print(Bcolors.ENDC)
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
		print(Bcolors.OKGREEN)
		print(sentence)
		print("")
		print(Bcolors.ENDC)

	def send_output(self, sentence):
		return sentence

class TelegramBot(Chatbot):
	def __init__(self):
		super().__init__()
		self.updater = Updater(token='2102452317:AAFNft6A_qjmYd4Od6p5Z0UNw1FzRmozr0g', use_context=True)
		self.dispatcher = self.updater.dispatcher

	def start(self, update: Update, context: CallbackContext):
		context.bot.send_message(chat_id = update.effective_chat.id, text="I am a bot, please talk to me")

	def echo(self, update: Update, context: CallbackContext):
		context.bot.send_message(chat_id = update.effective_chat.id, text = update.message.text)
		context.bot.send_message(chat_id = update.effective_chat.id, text = "Test")
		print("Test")

	def handle_input(self, update: Update, context: CallbackContext):
		
		sentence = update.message.text
		tag, prediction = self.process_input(sentence)
		output = self.generate_output(tag, prediction)
		context.bot.send_message(chat_id = update.effective_chat.id, text = output)


	def run(self):
		start_handler = CommandHandler('start', self.start)
		echo_handler = MessageHandler(Filters.text & (~Filters.command), self.echo)
		input_handler = MessageHandler(Filters.text & (~Filters.command), self.handle_input)
		self.dispatcher.add_handler(start_handler)
#		self.dispatcher.add_handler(echo_handler)
		self.dispatcher.add_handler(input_handler)

		self.updater.start_polling()


if __name__ == "__main__":
	bot = Chatbot()
	bot.runConsole()