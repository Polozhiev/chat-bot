from collections import defaultdict

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from params import Params
from ai import AI

params = Params() 

bot = Bot(token=...)
dp = Dispatcher(bot=bot)

HISTORY = defaultdict(list) 

model = AI() 

kb = [[types.KeyboardButton(text='Очистить историю'),],]
keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True, keyboard=kb)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.answer("Привет! Я немного глуповат, но умею болтать :)")

@dp.message_handler(text=['Очистить историю'])
async def clean_context(message: types.Message):
    HISTORY[message.from_user.id] = []
    await message.answer("История диалога очищена")

@dp.message_handler(content_types=['text'])
async def continue_dialog(message: types.Message):
    user = message.from_user.id
    HISTORY[user].append(message.text)

    dialog = HISTORY[user][-3:]
    response = model.answer(dialog, params)

    HISTORY[user].append(response)
    await message.answer(response, reply_markup=keyboard)

if __name__ == "__main__":
    executor.start_polling(dp)
