"""
Шаблон платного бота Telegram (aiogram 3) с оплатой Telegram Stars.

Что здесь заложено ради честного отношения к покупателю:
- цена и срок доступа показываются ДО оплаты;
- доступ выдаётся сразу после оплаты;
- /paysupport, /support, /terms, /status, /delete работают;
- владелец может вернуть Stars командой /refund;
- повторная доставка одного платежа невозможна (идемпотентность по charge_id).

Запуск:
    pip install aiogram
    export BOT_TOKEN=...   # токен от @BotFather
    export ADMIN_ID=...    # ваш Telegram user id
    python bot_template.py

Перед запуском в проде: замените тексты в блоке TEXTS, функцию deliver()
и проверьте оплату на минимальной сумме.
"""
import asyncio
import os
import sqlite3
import time

from aiogram import Bot, Dispatcher
from aiogram.filters import Command, CommandObject
from aiogram.types import LabeledPrice, Message, PreCheckoutQuery

TOKEN = os.environ["BOT_TOKEN"]
ADMIN_ID = int(os.environ["ADMIN_ID"])
PRICE_STARS = int(os.getenv("PRICE_STARS", "100"))
ACCESS_DAYS = int(os.getenv("ACCESS_DAYS", "30"))
SUPPORT_CONTACT = os.getenv("SUPPORT_CONTACT", "@your_support")

TEXTS = {
    "start": (
        "Привет! Здесь вы получаете: <что даёт продукт>.\n\n"
        f"Цена: {PRICE_STARS} Stars за {ACCESS_DAYS} дней доступа. "
        "Это разовая оплата, автоматического продления нет.\n"
        "Купить: /buy. Условия: /terms. Проблема с оплатой: /paysupport."
    ),
    "terms": (
        "Условия:\n"
        f"• Оплата {PRICE_STARS} Stars даёт доступ на {ACCESS_DAYS} дней.\n"
        "• Автопродления нет, вы сами решаете, продлевать ли.\n"
        "• Если доступ не выдан или не работает как описано, мы вернём Stars.\n"
        "• Храним только ваш Telegram ID, дату оплаты и срок доступа. "
        "Данные не передаём и не продаём. Удалить данные: /delete.\n"
        f"• Поддержка: {SUPPORT_CONTACT}, отвечаем в течение суток."
    ),
    "paysupport": (
        "Проблема с оплатой? Напишите нам: "
        f"{SUPPORT_CONTACT}. Приложите дату платежа. "
        "Если Stars списались, а доступ не выдан, мы вернём их."
    ),
    "support": f"Вопросы по работе бота: {SUPPORT_CONTACT}. Отвечаем в течение суток.",
}

db = sqlite3.connect("bot.db")
db.execute(
    "CREATE TABLE IF NOT EXISTS payments("
    "charge_id TEXT PRIMARY KEY, user_id INTEGER, stars INTEGER, "
    "paid_at INTEGER, refunded INTEGER DEFAULT 0)"
)
db.execute("CREATE TABLE IF NOT EXISTS access(user_id INTEGER PRIMARY KEY, expires_at INTEGER)")
db.commit()

bot = Bot(TOKEN)
dp = Dispatcher()


async def deliver(user_id: int) -> None:
    """Выдача продукта после оплаты: замените на свою логику (ссылка, файл, доступ)."""
    await bot.send_message(user_id, "Доступ открыт. Вот ваш материал: <ссылка или файл>")


@dp.message(Command("start"))
async def cmd_start(m: Message):
    await m.answer(TEXTS["start"])


@dp.message(Command("terms"))
async def cmd_terms(m: Message):
    await m.answer(TEXTS["terms"])


@dp.message(Command("paysupport"))
async def cmd_paysupport(m: Message):
    await m.answer(TEXTS["paysupport"])


@dp.message(Command("support"))
async def cmd_support(m: Message):
    await m.answer(TEXTS["support"])


@dp.message(Command("status"))
async def cmd_status(m: Message):
    row = db.execute("SELECT expires_at FROM access WHERE user_id=?", (m.from_user.id,)).fetchone()
    if row and row[0] > time.time():
        left = int((row[0] - time.time()) // 86400)
        await m.answer(f"Доступ активен, осталось примерно {left} дн. Продлить: /buy")
    else:
        await m.answer("Активного доступа нет. Купить: /buy")


@dp.message(Command("buy"))
async def cmd_buy(m: Message):
    await m.answer_invoice(
        title="Доступ на %d дней" % ACCESS_DAYS,
        description="<что именно получает покупатель>. Разовая оплата, без автопродления.",
        payload="access_%d" % ACCESS_DAYS,
        currency="XTR",  # Telegram Stars
        prices=[LabeledPrice(label="Доступ", amount=PRICE_STARS)],
        provider_token="",  # для Stars токен провайдера пустой
    )


@dp.pre_checkout_query()
async def pre_checkout(q: PreCheckoutQuery):
    # Отвечать нужно быстро (около 10 секунд). Здесь можно проверить, что товар доступен.
    await q.answer(ok=True)


@dp.message(lambda m: m.successful_payment is not None)
async def paid(m: Message):
    p = m.successful_payment
    cur = db.execute(
        "INSERT OR IGNORE INTO payments(charge_id, user_id, stars, paid_at) VALUES(?,?,?,?)",
        (p.telegram_payment_charge_id, m.from_user.id, p.total_amount, int(time.time())),
    )
    if cur.rowcount == 0:  # этот платёж уже обработан
        return
    row = db.execute("SELECT expires_at FROM access WHERE user_id=?", (m.from_user.id,)).fetchone()
    start = max(time.time(), row[0]) if row else time.time()
    db.execute(
        "INSERT OR REPLACE INTO access(user_id, expires_at) VALUES(?,?)",
        (m.from_user.id, int(start + ACCESS_DAYS * 86400)),
    )
    db.commit()
    await m.answer("Спасибо за оплату! Если что-то пойдёт не так, /paysupport.")
    await deliver(m.from_user.id)


@dp.message(Command("refund"))
async def cmd_refund(m: Message, command: CommandObject):
    """Только для владельца: /refund <user_id> <charge_id>"""
    if m.from_user.id != ADMIN_ID:
        return
    try:
        user_id_s, charge_id = (command.args or "").split()
        user_id = int(user_id_s)
    except ValueError:
        await m.answer("Формат: /refund <user_id> <charge_id>")
        return
    await bot.refund_star_payment(user_id=user_id, telegram_payment_charge_id=charge_id)
    db.execute("UPDATE payments SET refunded=1 WHERE charge_id=?", (charge_id,))
    db.execute("DELETE FROM access WHERE user_id=?", (user_id,))
    db.commit()
    await m.answer("Возврат выполнен.")
    await bot.send_message(user_id, "Мы вернули вам Stars. Приносим извинения за неудобства.")


@dp.message(Command("delete"))
async def cmd_delete(m: Message):
    # Записи о платежах оставляем для учёта (требование закона), остальное удаляем.
    db.execute("DELETE FROM access WHERE user_id=?", (m.from_user.id,))
    db.commit()
    await m.answer("Данные о вашем доступе удалены. Записи о платежах хранятся только для учёта.")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
