import re
import unittest

# Regex pattern to remove punctuation and special characters while keeping scripts
patterns = {
    "latin": re.compile(r"[^a-zA-Z\u00C0-\u017F\u0259\s]+"),
    "cyrillic": re.compile(r"[^а-яА-ЯёЁ\u0400-\u04FF\s]+"),
    "devanagari": re.compile(r"[^\u0900-\u0963\u0970-\u097F\s]+"),
    "perso_arabic": re.compile(r"[^\u0600-\u060b\u060e-\u06d3\u06d5-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDCF\uFDF0-\uFDFF\uFE70-\uFEFF\u061B\u061F\s]"),
    "japanese_kana": re.compile(r"[^\w\s぀-ヿ]"),
    "bengali": re.compile(r"[^\u0980-\u09FF\s]"),
}

# Language and script mapping
language_script_map = {
    "as": "bengali", # Assamese
    "ast": "latin", # Asturian
    "az": "latin", # Azerbaijani
    "ba": "cyrillic", # Bashkir
    "be": "cyrillic", # Belarusian
    "bg": "cyrillic", # Bulgarian
    "bn": "bengali", # Bengali
    "cs": "latin", # Czech
    "en": "latin", # English
    "es": "latin", # Spanish
    "fy-NL": "latin", # West Frisian
    "hi": "devanagari", # Hindi
    "ja": "japanese_kana", # Japanese
    "kk": "cyrillic", # Kazakh
    "ky": "cyrillic", # Kyrgyz
    "mk": "macedonian", # Macedonian
    "nl": "latin", # Dutch
    "ru": "cyrillic", # Russian
    "sk": "latin", # Slovak
    "tt": "cyrillic", # Tatar
    "ug": "perso_arabic", # Uyghur
    "ur": "perso_arabic", # Urdu
    "uz": "latin", # Uzbek
}

def normalize_text(text: str,
                   lang: str) -> str:
    """Normalize text by removing punctuation and special characters."""
    text = text.lower()
    script = language_script_map[lang]
    if script in patterns:
        return patterns[script].sub("", text)
    else:
        raise ValueError(f"Unsupported script: {script}")


# Test cases
texts = {
    "latin": [
        ("Hello, world!", "Hello world"),
        ("Python@3.9 is great!", "Python is great"),
        ("Let's test: this; function.", "Lets test this function"),
    ],
    "cyrillic": [
        ("Привет, мир!", "Привет мир"),
        ("Это тест: проверка; функции.", "Это тест проверка функции"),
    ],
    "devanagari": [
        ("नमस्ते, दुनिया!", "नमस्ते दुनिया"),
        ("यह एक परीक्षण है: देखें; क्या होता है।", "यह एक परीक्षण है देखें क्या होता है"),
    ],
    "perso_arabic": [
        ("سلام، دنیا!", "سلام دنیا"),
        ("این یک آزمایش است: ببینیم چه میشود.", "این یک آزمایش است ببینیم چه میشود"),
    ],
    "japanese_kana": [
        ("こんにちは、世界！", "こんにちは世界"),
        ("テスト：これは、ちゃんと動く？", "テストこれはちゃんと動く"),
    ],
    "bengali": [
        ("হ্যালো, বিশ্ব!", "হ্যালো বিশ্ব"),
        ("এটি একটি পরীক্ষা: দেখে নিন; এটি কাজ করে কিনা।", "এটি একটি পরীক্ষা দেখে নিন এটি কাজ করে কিনা"),
    ],
}


class TestRegexCleaning(unittest.TestCase):
    def test_cleaning(self):
        for script, cases in texts.items():
            for input_text, expected_output in cases:
                with self.subTest(script=script, input_text=input_text):
                    cleaned_text = patterns[script].sub("", input_text)
                    self.assertEqual(cleaned_text, expected_output)


if __name__ == "__main__":
    unittest.main()