
from telegram import Update
from chatbot import Chatbot, TelegramBot
from trainer import Trainer

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



def start():
	print(Bcolors.WARNING)
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
	print(Bcolors.OKGREEN)
	user_input = input("Select: ")
	print(Bcolors.ENDC)

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
	cb.runConsole()

def retrain():
	trainer = Trainer()
	trainer.run()

if __name__ == "__main__":
	start()