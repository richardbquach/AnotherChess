import pygame

class SoundMixer:
    sounds = []

    def __init__(self):
        self.sounds.append(pygame.mixer.Sound("src/assets/sounds/capture.mp3"))
        self.sounds.append(pygame.mixer.Sound("src/assets/sounds/move-self.mp3"))

    def playMove(self, capture, check):
        if check:
            self.sounds[0].play(1)
        elif capture:
            self.sounds[0].play()
        else:
            self.sounds[1].play()