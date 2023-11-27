import torch
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import os
import torch.nn.functional as F
import numpy as np
import cv2
from copy import deepcopy


def Phrase2idx(prompt, phrases):
    def match(prompt_words: list[str], phrase_words: list[str]):
        if prompt_words == phrase_words:
            return True
        for prompt_word, phrase_word in zip(prompt_words, phrase_words):
            if prompt_word != phrase_word and prompt_word != phrase_word+'s' and prompt_word != phrase_word+'es':
                return False
        return True
    phrases = [x.replace('_', ' ') for x in phrases.split(';')]
    object_positions = []
    for punc in [',', '.', ';', ':', '?', '!']:
        prompt = prompt.replace(punc, ' '+punc)
    words = prompt.split()  

    for phrase in phrases:
        phrase_words = phrase.split()  
        positions = []

        for i in range(len(words) - len(phrase_words) + 1):
            if match(words[i:i + len(phrase_words)], phrase_words):
                positions += list(range(i+1, i + len(phrase_words)+1))
        if positions == []:
            print(prompt)
            print(phrases)
            return None
        object_positions.append(positions)

    return object_positions



def setup_logger(save_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(os.path.join(save_path, f"{logger_name}.log"))
    file_handler.setLevel(logging.INFO)

    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def save_image(pil_img, bboxes, phrases, save_path):
    pil_img.save(save_path)
    
    

def draw_box(pil_img, bboxes, phrases, save_path):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype('./FreeMono.ttf', 25)
    phrases = [x.strip() for x in phrases.split(';')]
    for obj_bboxes, phrase in zip(bboxes, phrases):
        for obj_bbox in obj_bboxes:
            x_0, y_0, x_1, y_1 = obj_bbox[0], obj_bbox[1], obj_bbox[2], obj_bbox[3]
            draw.rectangle([int(x_0 * 512), int(y_0 * 512), int(x_1 * 512), int(y_1 * 512)], outline='red', width=5)
            draw.text((int(x_0 * 512) + 5, int(y_0 * 512) + 5), phrase, font=font, fill=(255, 0, 0))
    # pil_img.save(save_path)