import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import numpy as np
import pandas as pd
import pyscreenshot as ImageGrab #grabbing image
import cv2
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
        # Opening Chrome to run the game
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome("/home/fake_batman_/PycharmProjects/Chrome-Dino-AI/chromedriver", chrome_options=chrome_options)
        self._driver.set_window_size(200, 300)
        self._driver.set_window_position(-10, 0)
        self._driver.get("file:///home/fake_batman_/PycharmProjects/Chrome-Dino-AI/game/dino.html")
        if custom_config:
            self._driver.execute_script("Runner.config.ACCELERATION=0")

    # These are the functions that act as an interface between Python and JavaScript

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


# For game environment
class GameState:
    def __init__(self, agent, game):
        self._game = game
        self._agent = agent
        self._display = show_img()
        self._display.__next__()

    def get_state(self, actions):
        actions_df.loc[len(actions_df)] = actions[1]
        score = self._game.get_score()
        reward = 0.1*score/10
        is_over = False
        if actions[1] == 1:
            self._agent.jump()
            reward = 0.1*reward/11
        image = grab_screen()
        self._display.send(image)

        if self._agent.is_crashed():
            scores_df.loc[len(loss_df)] = score
            self._game.restart()
            reward = -11/score
            is_over = True
        return image, reward, is_over


def grab_screen(_driver=None):
    screen = np.array(ImageGrab.grab(bbox=(40, 180, 440, 400)))
    image = process_img(screen)
    return image


def process_img(image):
    image = cv2.resize(image, (0, 0), fx=0.15, fy=0.10)
    image = image[2:38, 10:50]
    image = cv2.Canny(image, threshold1=100, threshold2=200)
    return image


def save_obj(obj, name):
    with open('objects/' + name + '.pkl', 'wb+') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('objects/' + name + '.pkl', 'rb') as file:
        return pickle.load(file)


def show_img(graphs=False):
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


loss_file_path = "./objects/loss_df.csv"
actions_file_path = "./objects/actions_df.csv"
scores_file_path = "./objects/scores_df.csv"
loss_df = pd.read_csv(loss_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns=['loss'])
scores_df = pd.read_csv(scores_file_path) if os.path.isfile(loss_file_path) else pd.DataFrame(columns=['scores'])
actions_df = pd.read_csv(actions_file_path) if os.path.isfile(actions_file_path) else pd.DataFrame(columns=['actions'])
LEARNING_RATE = 1e-4
img_rows, img_cols = 40, 20
img_channels = 4
ACTIONS = 2
OBSERVATION = 50000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
EXPLORE = 100000
REPLAY_MEMORY = 50000
BATCH = 32
GAMMA = 0.99


# Only done once, after that it is commented
# PS: Could use a simple if-else to check if these objects exists.
def init_cache():
    save_obj(INITIAL_EPSILON, "epsilon")
    t = 0
    save_obj(t, "time")
    D = deque()
    save_obj(D, "D")


# Simple Model
def build_model():
    print("Building model")
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
    print("Model built")
    return model


def train_network(model, game_state):
    D = load_obj("D")
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    # get the first state by doing nothing
    x_t, r_0, terminal = game_state.get_state(do_nothing)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2).reshape(1, 20, 40, 4)  # stacking 4 images
    # load observe, epsilon, timespace, model
    OBSERVE = OBSERVATION
    epsilon = load_obj("epsilon")
    t = load_obj("time")
    model.load_weights("model_final.h5")
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=adam)
    # initial state is set to first state obtained by doing nothing
    initial_state = s_t
    while True:
        # at t = 0
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        if t % 1 == 0:
            if random.random() <= epsilon:  # Exploring by doing random actions
                print("---Random Action---")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)  # Predict
                max_Q = np.argmax(q)  # Choose the out of all the Q-values
                action_index = max_Q
                a_t[action_index] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x_t1, r_t, terminal = game_state.get_state(a_t)  # get the next state after performing action a_t
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        D.append((s_t, action_index, r_t, s_t1, terminal))  # Adding the state for replay memory
        if len(D) > REPLAY_MEMORY:
            D.popleft()
        if t > OBSERVE:
            # Replay Memory
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32, 20, 40, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))  # 32, 2

            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                inputs[i:i+1] = state_t
                targets[i] = model.predict(state_t)
                Q_sa = model.predict(state_t1)
                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA*np.max(Q_sa)
            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
        else:
            time.sleep(0.12)
        s_t = initial_state if terminal else s_t1
        t = t + 1
        print("Timestep: {}\tEpsilon: {}\tAction: {}\tQ_max: {}\tLoss:\t{}\tReward: {}".format(t, epsilon, action_index,
                                                                                               np.max(Q_sa), loss, r_t))
        if t % 1000 == 0:
            print("Saving model...")
            model.save_weights("model_final.h5", overwrite=True)
            save_obj(D, "D")
            save_obj(t, "time")
            save_obj(epsilon, "epsilon")
            loss_df.to_csv("objects/loss_df.csv", index=False)
            scores_df.to_csv("objects/scores_df.csv", index=False)
            actions_df.to_csv("objects/actions_df.csv", index=False)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)


def play_game(observe=False):
    init_cache()
    game = Game()
    dino = DinoAgent(game)
    game_state = GameState(dino, game)
    model = build_model()
    try:
        train_network(model, game_state)
    except StopIteration:
        game.end()


play_game(observe=True)
