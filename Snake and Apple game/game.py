import pygame
import random

pygame.init()

screen = pygame.display.set_mode((700, 700))
pygame.display.set_caption('Snake and Apple Game')

score = 0

red = (255, 0, 0)
blue = (0, 0, 255)
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)

snake_head_x = 700 / 2
snake_head_y = 700 / 2
snake_head_width = 25
snake_head_height = 25
snake_head_speed = 7
snake_head_directiond = ''

apple_width = 15
apple_height = 15
apple_x = random.randint(0, 700 - apple_width)
apple_y = random.randint(0, 700 - apple_height)

font = pygame.font.Font(None, 36)

def update_score_text(score):
    return font.render(f'Score: {score}', True, white)

text = update_score_text(score)
text_rect = text.get_rect()
text_rect.center = (50, 50)

class SnakeBody:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

snake_body = []

fps = 60
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(green)

    snake_head = pygame.Rect(snake_head_x, snake_head_y, snake_head_width, snake_head_height)
    apple = pygame.Rect(apple_x, apple_y, apple_width, apple_height)

    pygame.draw.rect(screen, blue, snake_head)
    pygame.draw.rect(screen, red, apple)
    screen.blit(text, text_rect)

    keys = pygame.key.get_pressed()
    if (keys[pygame.K_LEFT] or keys[pygame.K_a]) and snake_head_direction != 'right':
        snake_head_direction = 'left'
    if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) and snake_head_direction != 'left':
        snake_head_direction = 'right'
    if (keys[pygame.K_UP] or keys[pygame.K_w]) and snake_head_direction != 'down':
        snake_head_direction = 'up'
    if (keys[pygame.K_DOWN] or keys[pygame.K_s]) and snake_head_direction != 'up':
        snake_head_direction = 'down'

    if snake_head_y > 700 or snake_head_y < 0 or snake_head_x > 700 or snake_head_x < 0:
        running = False

    if snake_head_direction == 'left':
        snake_head_x -= snake_head_speed
    if snake_head_direction == 'right':
        snake_head_x += snake_head_speed
    if snake_head_direction == 'up':
        snake_head_y -= snake_head_speed
    if snake_head_direction == 'down':
        snake_head_y += snake_head_speed

    if snake_head.colliderect(apple):
        score += 1
        text = update_score_text(score)
        apple_x = random.randint(0, 700 - apple_width)
        apple_y = random.randint(0, 700 - apple_height)
        snake_body.append(SnakeBody(snake_head_x, snake_head_y, snake_head_width, snake_head_height))

    if len(snake_body) > 0:
        for i in range(len(snake_body) - 1, 0, -1):
            snake_body[i].x = snake_body[i - 1].x
            snake_body[i].y = snake_body[i - 1].y

        snake_body[0].x = snake_head_x
        snake_body[0].y = snake_head_y

    for segment in snake_body:
        pygame.draw.rect(screen, blue, segment.get_rect())

    clock.tick(fps)
    pygame.display.update()

pygame.quit()
