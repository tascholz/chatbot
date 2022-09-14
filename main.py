from telegram.ext import Updater, CallbackContext, CommandHandler, MessageHandler, Filters
from telegram import Update
from chatbot import Chatbot
from trainer import Trainer


class TelegramBot():
	def __init__(self):


		self.updater = Updater(token='2102452317:AAFNft6A_qjmYd4Od6p5Z0UNw1FzRmozr0g', use_context=True)
		self.dispatcher = self.updater.dispatcher

	def start(self, update: Update, context: CallbackContext):
		context.bot.send_message(chat_id = update.effective_chat.id, text="I am a bot, please talk to me")

	def echo(self, update: Update, context: CallbackContext):
		context.bot.send_message(chat_id = update.effective_chat.id, text = update.message.text)


	def run(self):
		start_handler = CommandHandler('start', self.start)
		echo_handler = MessageHandler(Filters.text & (~Filters.command), self.echo)
		self.dispatcher.add_handler(start_handler)
		self.dispatcher.add_handler(echo_handler)

		self.updater.start_polling()


def start():
	print("")
	print("------------------------------------")
	print("|           Chatbot                |")
	print("|----------------------------------|")
	print("|Please select a mode to start with|")
	print("|--------------------------------- |")
	print("|0: Telegram Bot                   |")
	print("|1: Console                        |")
	print("|2: Retrain                        |")
	print("|q: Exit application               |")
	print("------------------------------------")
	print("")
	user_input = input("Select: ")

	if user_input == "0":
		telegram()
	elif user_input == "1":
		console()
	elif user_input == "2":
		retrain()
	elif user_input == "q":
		print("Exiting...")

def telegram():
	print("Running TelegramBot application")
	tb = TelegramBot()
	tb.run()

def console():
	print ("Running console application...")
	print("")
	cb = Chatbot()
	cb.run()

def retrain():
	trainer = Trainer()
	trainer.run()

if __name__ == "__main__":
	start()