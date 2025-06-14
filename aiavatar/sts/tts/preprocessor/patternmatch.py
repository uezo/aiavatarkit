import re
import logging
from typing import List, Tuple, Union, Callable
from . import TTSPreprocessor


class PatternMatchPreprocessor(TTSPreprocessor):
    def __init__(
        self,
        *,
        patterns: List[Union[Tuple[str, str], Tuple[re.Pattern, str]]] = None
    ):
        self.compiled_patterns = []
        if patterns:
            for pattern in patterns:
                self.add_pattern(pattern[0], pattern[1])

    def add_number_dash_pattern(self):
        num_map = {
            "0": "ゼロ ", "1": "イチ ", "2": "ニ ", "3": "サン ", "4": "ヨン ",
            "5": "ゴ ", "6": "ロク ", "7": "ナナ ", "8": "ハチ ", "9": "キュー "
        }

        def convert_numbers(match):
            # Split by dash and convert each number
            parts = match.group(0).split("-")
            converted_parts = []
            for part in parts:
                # Convert each digit
                converted_digits = "".join(num_map.get(digit, digit) for digit in part)
                converted_parts.append(converted_digits)
            return "の".join(converted_parts)
        
        self.add_pattern(r"\d+(?:-\d+)+", convert_numbers, regex=True)

    def add_phonenumber_pattern(self):
        num_map = {
            "0": "ゼロ ", "1": "イチ ", "2": "ニ ", "3": "サン ", "4": "ヨン ",
            "5": "ゴ ", "6": "ロク ", "7": "ナナ ", "8": "ハチ ", "9": "キュー "
        }
        
        def convert_phone(match):
            # Split by dash and convert each part
            parts = match.group(0).split("-")
            converted_parts = []
            for part in parts:
                # Convert each digit
                converted_digits = "".join(num_map.get(digit, digit) for digit in part)
                converted_parts.append(converted_digits)
            return "の".join(converted_parts)
        
        # Match phone number patterns: 090-1234-5678, 03-1234-5678, etc.
        self.add_pattern(r"\d{2,4}-\d{4}-\d{4}", convert_phone, regex=True)

    def add_pattern(self, pattern: Union[str, re.Pattern], replacement: Union[str, Callable], *, regex: bool = False):
        if isinstance(pattern, re.Pattern):
            compiled_pattern = pattern
        elif regex:
            compiled_pattern = re.compile(pattern)
        else:
            escaped_pattern = re.escape(pattern)
            compiled_pattern = re.compile(escaped_pattern)
        self.compiled_patterns.append((compiled_pattern, replacement))

    async def process(self, text: str, style_info: dict = None, language: str = None) -> str:
        converted = text
        for compiled_pattern, replacement in self.compiled_patterns:
            try:
                converted = compiled_pattern.sub(replacement, converted)
            except Exception as ex:
                logging.error(f"Error at pattern match (text={converted}): {ex}", exc_info=True)

        return converted
