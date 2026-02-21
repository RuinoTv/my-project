from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from datetime import datetime
import nltk
import re
import random
from nltk.stem.snowball import SnowballStemmer

#---------------------------------------------------
now = datetime.now()
year = now.year
week = now.isocalendar().week
day = now.isocalendar().weekday
vremz = f"год {year}, неделя {week}, день {day}"
#---------------------------------------------------

#отпечатканы жондитын жеры
stemmer = SnowballStemmer("russian")

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # стемминг
    words = text.split()
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

# база данных-1
database = [
    {
        "question": ["название университета",
                     "имя университета",
                     "имя вуза",
                     "название вуза"],
        "answer": ["ALT", "Алит", "АЛТ"]
    },
    {
        "question": ["сколько блоков в университете"],
        "answer": ["5", "пять", "бес"]
    },
    {
        "question": ["где находится университет ALT",
                     "местоположение университета",
                     "где расположен университет"],
        "answer": ["Шевченко 97 / Масанчи 71"]
    },
    {
        "question": ["какой сейчас год",
                     "текущий год",
                     "что за год у нас сейчас",
                     "сейчас какой год"],
        "answer": [vremz]
    },
    {
        "question": ["как тебя зовут",
                     "какое твое имя",
                     "как тебя звать"],
        "answer": ["UA"]
    },
    {
        "question": ["где находится 5 блок",
                     "как найти 5 блок",
                     "в какой зоне находится 5 блока",
                     "где находится 5 корпус",
                     "как найти 5 корпус",
                     "в какой зоне находится 5 корпус"],
        "answer": ["Пятый корпус расположен в оранжевой зоне. Чтобы попасть туда, нужно пройти через синюю зону. На 2-6 этажах поверните к северу и следуйте до конца коридора. В конце поверните налево."]
    },
    {
        "question": ["где находится 4 блок",
                     "как найти 4 блок",
                     "в какой зоне находится 4 блока",
                     "где находится 4 корпус",
                     "как найти 4 корпус",
                     "в какой зоне находится 4 корпус"],
        "answer": ["Четвёртый корпус расположен в красной зоне. Чтобы попасть туда, нужно пройти через оранжевую зону. На 2 этаже идите почти до конца коридора. И почти в конце поверните налево."]
    },
    {
        "question": ["где находится 3 блок",
                     "как найти 3 блок",
                     "в какой зоне находится 3 блока",
                     "где находится 3 корпус",
                     "как найти 3 корпус",
                     "в какой зоне находится 3 корпус"],
        "answer": ["Третий корпус расположен в зеленой зоне. Чтобы попасть туда, нужно пройти через синюю зону. На 2-3 этажах поверните к югу и следуйте до первого поворота на право и поверните направо, и вы попадёте в колледж, это и есть 3 корпус."]
    },
    {
        "question": ["где находится 2 блок",
                     "как найти 2 блок",
                     "в какой зоне находится 2 блока",
                     "где находится 2 корпус",
                     "как найти 2 корпус",
                     "в какой зоне находится 2 корпус"],
        "answer": ["Второй корпус находится в синей зоне. Чтобы попасть туда, необходимо пройти через турникет в главном корпусе и подняться на второй этаж.."]
    },
    {
        "question": [
            "где находится зона отдыха",
            "как найти зону отдыха",
            "в какой зоне находится зона отдыха",
            "где расположен зона отдыха"],
        "answer": ["Зона отдыха расположена на втором этаже главного корпуса."]
    },
{
        "question": [
            "где находится библиотека",
            "как найти библиотеку",
            "в какой зоне находится библиотека",
            "где расположен библиотека"],
        "answer": ["Чтобы попасть в библиотеку, необходимо на первом этаже главного корпуса пройти через турникет. По дороге к лестнице поверните направо. Вы увидите надпись «Coworking», которая указывает на расположение библиотеки."]
    },
    {
        "question": ["где находится деканат",
                     "где расположен деканат",
                     "где находится декан",
                     "где расположен декан"],
        "answer": ["Аудитория В305"]
    }
]

# ---- NLP ----
questions = [normalize(q) for item in database for q in item["question"]]

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),   #быр немесе бырнеше соз
    stop_words=None
)

tfidf_matrix = vectorizer.fit_transform(questions)

#база данных-2
def get_room_location(user_question):
    building_words = {
        "В": "втором",
        "G": "третьем",
        "R": "четвёртом",
        "O": "пятом"
    }

    floor_words = {
        "1": "первый",
        "2": "второй",
        "3": "третий",
        "4": "четвёртый",
        "5": "пятый",
        "6": "шестой",
        "7": "седмой",
        "8": "восьмой"
    }

    # база данных-2 ыздейды
    matches = re.findall(r'\b([А-ЯA-Z])(\d+)\b', user_question.upper())
    if not matches:
        return None

    answers = []
    for match in matches:
        building_letter = match[0]
        floor_digit = match[1][0]
        room_code = f"{building_letter}{match[1]}"

        building_word = building_words.get(building_letter, building_letter)
        floor_word = floor_words.get(floor_digit, floor_digit)

        answers.append(
            f"{room_code} аудитория находится на {building_word} корпусе, {floor_word} этаж."
        )

    return " ".join(answers)


# ---- чат ----
user_input = input("Сұрақ қой: ")
user_question = normalize(user_input)

# суракта аудитория сурадыма сурамадыма ыздейды
dynamic_answer = get_room_location(user_input)
if dynamic_answer:
    print(dynamic_answer)
else:
    # аудитория туралы сурак жок болса база данных-1 ден суракты ыздейды
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, tfidf_matrix)[0]
    best_index = similarities.argmax()
    best_score = similarities[best_index]

    # крч қай база данных қай формулировкаға жатады соны іздейді
    cumulative_index = 0
    for item in database:
        if best_index < cumulative_index + len(item["question"]):
            print(random.choice(item["answer"]))
            break
        cumulative_index += len(item["question"])
    else:
        print("вопрос задан некоректно")