from asyncio.log import logger
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# アクセストークン（先ほど発行されたアクセストークンに書き換えてください）
TOKEN = os.environ.get("TOKEN")


class TelegramBot:
    def __init__(self, system):
        # systemを変更することで挙動を変えられる
        self.system = system

    # WARN: python-telegram-bot==12.8でないと動かない
    def start(self, bot, update):
        # 辞書型 inputにユーザIDを設定
        input = {"utt": None, "sessionId": str(update.message.from_user.id)}

        # システムからの最初の発話をinitial_messageから取得し，送信
        update.message.reply_text(self.system.initial_message(input)["utt"])

    def message(self, bot, update):
        # 辞書型 inputにユーザからの発話とユーザIDを設定
        input = {
            "utt": update.message.text,
            "sessionId": str(update.message.from_user.id),
        }

        # replyメソッドによりinputから発話を生成
        system_output = self.system.reply(input)

        # 発話を送信
        update.message.reply_text(system_output["utt"])

    def run(self):
        updater = Updater(TOKEN)
        dp = updater.dispatcher
        dp.add_handler(CommandHandler("start", self.start))
        dp.add_handler(MessageHandler(Filters.text, self.message))
        updater.start_polling()
        updater.idle()
