
import pyautogui
import time
import random


while True: 
           x= random.randint(100,600)

           y=random.randint(100,600)

pyautogui.moveto(x,y,duration=8.5)

time.sleep(5)
