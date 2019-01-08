#!pip install python-telegram-bot --upgrade
# # https://api.telegram.org/bot<773551646:AAFz2wukaMVrpufXZyaBctoo7dxyVlwxFtM>/getUpdates

chat_id = ''
token = ''

def sendNotification(added_info = ''):
    """
    send the message in our telegram chanel

    added_info: str, added text in message
    """
    from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
    updater = Updater(token=token) # Токен API к боту Elephantbot
    dispatcher = updater.dispatcher
    bot = updater.bot
    bot.send_message(chat_id=chat_id,text=added_info) # Номер нашего канала

