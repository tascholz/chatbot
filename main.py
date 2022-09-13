from telegram.ext import Updater, CallbackContext, CommandHandler, MessageHandler, Filters
from telegram import Update


updater = Updater(token='2102452317:AAFNft6A_qjmYd4Od6p5Z0UNw1FzRmozr0g', use_context=True)
dispatcher = updater.dispatcher

def start(update: Update, context: CallbackContext):
	context.bot.send_message(chat_id = update.effective_chat.id, text="I am a bot, please talk to me")

def echo(update: Update, context: CallbackContext):
	context.bot.send_message(chat_id = update.effective_chat.id, text = update.message.text)


def run():
	start_handler = CommandHandler('start', start)
	echo_handler = MessageHandler(Filters.text & (~Filters.command), echo)
	dispatcher.add_handler(start_handler)
	dispatcher.add_handler(echo_handler)

	updater.start_polling()

if __name__ == "__main__":
	run()