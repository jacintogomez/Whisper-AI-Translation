import pygame
pygame.init()

#sound=pygame.mixer.Sound('recordings/test.wav')
sound=pygame.mixer.Sound('recordings/span.mp3')
sound.play()

pygame.time.wait(5000)
pygame.quit()