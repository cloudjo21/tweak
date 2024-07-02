import re


def normalize_digit(text):
    return re.sub(r'\d', '0', text)
