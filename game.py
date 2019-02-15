import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import numpy as np
import pandas as pd
from PIL import ImageGrab #grabbing image
from PIL import Image
import cv2 #opencv
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
from keras.callbacks import TensorBoard
from collections import deque
import random
import pickle
import json


class Game:
    def __init__(self, custom_config=True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(chrome_options=chrome_options)
        self._driver.set_window_position(-10, 0)
        self._driver.set_window_size(200, 300)
        self._driver.get(os.path.abspath("game/dino.html"))
        if custom_config:
            self._driver.execute_script("Runner.config.ACCELERATION=0")

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("return Runner.instance_.restart()")
        time.sleep(0.25)

    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()


class DinoAgent:
    def __init__(self, game):
        self._game = game
        self.jump() # to start the game we have to jump once
        time.sleep(.5)

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()


class GameState:
    def __init__(self, agent, game):
        self._game = game
        self_agent = agent

    def get_state(self, actions):
        score = self._game.get_score()
        reward = 0.1*score/10
        is_over = False
        if actions[1] == 1:
            self._agent.jump()
            reward = 0.1*reward/11
        image = grab_screen()

        if self._agent.is_crashed():
            self._game.restart()
            reward = -11/score
            is_over = True


def grab_screen(_driver = None):
    screen = np.array(ImageGrab.grab(bbox=(40, 180, 440, 400)))
    image = process_img(screen)
    return image


def process_img(image):
    image = cv2.resize(image, (0, 0), fx=0.15, fy=0.10)
    image = image[2:38, 10:50]
    image = cv2.Canny(image, threshold1=100, threshold2=200)
    return image


LEARNING_RATE = 1e-4
img_rows, img_cols = 40, 20
img_channels = 4
ACTIONS = 2


def build_model():
    print("Now we build the model")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same', input_shape=(img_cols, img_rows, img_channels)))  # 20*40*4
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(ACTIONS))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    print("We finish building the model")
    return model


