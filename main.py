import speech_recognition as sr
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import random
import string
import time


class VoiceRecognizer:
    def __init__(self, duration=1, volume_threshold=500, similarity_threshold=0.8):
        """
        :param duration: длительность прослушивания одного фрагмента в секундах
        :param volume_threshold: порог громкости для определения наличия голоса
        :param similarity_threshold: порог похожести для распознавания знакомого голоса
        """
        self.voice_db = {}  # База данных голосов: {имя: MFCC-вектор}
        self.duration = duration
        self.volume_threshold = volume_threshold
        self.similarity_threshold = similarity_threshold
        self.recognizer = sr.Recognizer()
        # Задаём sample_rate=16000 для стабильности обработки (если поддерживается)
        self.microphone = sr.Microphone(sample_rate=16000)

        # Предварительная настройка микрофона для учета фонового шума
        with self.microphone as source:
            print("[LOG] Настраиваюсь на окружающий шум...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("[LOG] Микрофон настроен.")

    def generate_random_name(self, length=6):
        """Генерирует случайное имя из букв и цифр"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    def extract_features(self, audio_data):
        """
        Извлекает усредненные MFCC-признаки из массива аудиоданных.
        Если длина аудио меньше 16000 сэмплов (примерно 1 секунда при 16kHz),
        функция дополняет сигнал до нужной длины.
        """
        try:
            # Приводим аудио к фиксированной длине в 16000 сэмплов
            y = librosa.util.fix_length(audio_data, size=16000)
            # Извлекаем MFCC (например, 13 коэффициентов)
            mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
            # Усредняем по времени – получаем вектор признаков
            return np.mean(mfcc, axis=1)
        except Exception as e:
            print(f"[LOG] Ошибка при извлечении признаков: {e}")
            return None

    def process_audio(self, audio):
        """
        Преобразует аудиоданные в numpy-массив, вычисляет уровень громкости.
        Если уровень ниже порога, считается, что голос не найден.
        Если голос найден – извлекаются признаки.
        """
        try:
            # Получаем сырые данные и преобразуем в float32
            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32)
            # Вычисляем RMS (корень из среднего квадрата)
            volume = np.sqrt(np.mean(audio_data ** 2))
            if volume < self.volume_threshold:
                print("[LOG] Голос не найден (низкий уровень звука).")
                return None
            else:
                print(f"[LOG] Голос найден. Уровень громкости: {volume:.2f}")
                features = self.extract_features(audio_data)
                return features
        except Exception as e:
            print(f"[LOG] Ошибка обработки аудио: {e}")
            return None

    def compare_voice(self, features):
        """
        Сравнивает извлечённые признаки с базой голосов.
        Возвращает кортеж (user_id, similarity), где similarity – наибольшая похожесть.
        Если база пуста – возвращает (None, -1).
        """
        best_match = None
        best_score = -1
        for user_id, saved_features in self.voice_db.items():
            similarity = cosine_similarity([features], [saved_features])[0][0]
            if similarity > best_score:
                best_score = similarity
                best_match = user_id
        return best_match, best_score

    def start_introduction(self):
        """Режим знакомства с пользователями"""
        print("[LOG] Начинаем процесс знакомства.")
        while True:
            user_name = input(
                "[LOG] Введите имя для нового пользователя (или 'стоп' для завершения знакомства): ").strip()
            if user_name.lower() == 'стоп':
                print("[LOG] Процесс знакомства завершён.")
                break

            print(f"[LOG] Пожалуйста, произнесите фразу: 'Привет, это мой голос.'")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)

            features = self.process_audio(audio)
            if features is not None:
                self.voice_db[user_name] = features
                print(f"[LOG] Голос пользователя '{user_name}' успешно добавлен в базу.")
            else:
                print("[LOG] Голос не был распознан, попробуйте снова.")

    def run(self):
        print("[LOG] Запуск режима реального времени. Слушаю микрофон...")
        while True:
            start_time = time.time()
            try:
                # Прослушиваем микрофон: ждем не более self.duration секунд для фразы
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=self.duration, phrase_time_limit=self.duration)
            except sr.WaitTimeoutError:
                print("[LOG] Тайм-аут. Голос не найден.")
                time.sleep(self.duration)  # Ждем self.duration секунд перед следующей итерацией
                continue

            features = self.process_audio(audio)
            if features is None:
                # Если голос не обнаружен или не удалось извлечь признаки, переходим к следующей итерации
                elapsed = time.time() - start_time
                if elapsed < self.duration:
                    time.sleep(self.duration - elapsed)
                continue

            # Для отладки выводим извлечённые признаки
            print(f"[DEBUG] Извлечённые признаки: {features}")

            # Сравниваем голос с базой
            user_id, score = self.compare_voice(features)
            if score >= self.similarity_threshold:
                print(f"[LOG] Распознан голос: {user_id} (similarity: {score:.2f})")
            else:
                print("[LOG] Новый голос, распознавание не удалось.")
            # Обеспечиваем, что итерация длится примерно self.duration секунд
            elapsed = time.time() - start_time
            if elapsed < self.duration:
                time.sleep(self.duration - elapsed)


if __name__ == '__main__':
    recognizer = VoiceRecognizer(duration=1, volume_threshold=500, similarity_threshold=0.8)
    recognizer.start_introduction()  # Вначале начинается знакомство
    recognizer.run()  # После знакомства запускается распознавание
