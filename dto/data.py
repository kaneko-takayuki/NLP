# -*- coding: utf-8 -*-


class ParseResult:
    def __init__(self, surface: str, original: str):
        self.surface = surface
        self.original = original


class Word:
    def __init__(self, word: str, vector: list):
        self.word = word
        self.vector = vector


class Phrase:
    def __init__(self, phrase: str, vector: list):
        self.phrase = phrase
        self.vector = vector


class Data:
    def __init__(self, label: int, sentence: str, phrases: list):
        self.label = label
        self.sentence = sentence
        self.phrases = phrases


class LearningData:
    def __init__(self, label: int, vector: list):
        self.label = label
        self.vector = vector
