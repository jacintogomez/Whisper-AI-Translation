import pygame
pygame.init()

#sound=pygame.mixer.Sound('recordings/test.wav')
sound=pygame.mixer.Sound('recordings/span.mp3')
sound.play()

delay=int(sound.get_length()*1000)
pygame.time.wait(delay)
pygame.quit()