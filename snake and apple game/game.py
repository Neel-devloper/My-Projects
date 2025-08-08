import pygame
import random

pygame.init()

screen = pygame.display.set_mode((700, 700))
pygame.display.set_caption('Snake and Apple Game')

red = (255, 0, 0)
blue = (0, 0, 255)
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
gray = (100, 100, 100)

font = pygame.font.Font(None, 36)
big_font = pygame.font.Font(None, 72)

box_size = 25

def update_score_text(score):
    return font.render(f'Score: {score}', True, white)

class SnakeBody:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

def reset_game():
    snake_head_x = 700 / 2
    snake_head_y = 700 / 2
    snake_head_width = 25
    snake_head_height = 25
    snake_head_speed = 7
    snake_head_direction = ''

    apple_width = 15
    apple_height = 15
    apple_x = random.randint(0, (700 - box_size) // box_size) * box_size + (box_size - apple_width) // 2
    apple_y = random.randint(0, (700 - box_size) // box_size) * box_size + (box_size - apple_height) // 2

    score = 0
    snake_body = []

    text = update_score_text(score)
    text_rect = text.get_rect()
    text_rect.center = (50, 50)

    return (snake_head_x, snake_head_y, snake_head_width, snake_head_height, snake_head_speed, snake_head_direction,
            apple_x, apple_y, apple_width, apple_height, score, snake_body, text, text_rect)

def draw_button(text, rect, color, hover_color):
    mouse_pos = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if rect.collidepoint(mouse_pos):
        pygame.draw.rect(screen, hover_color, rect)
        if click[0] == 1:
            return True
    else:
        pygame.draw.rect(screen, color, rect)

    button_text = font.render(text, True, white)
    button_text_rect = button_text.get_rect(center=rect.center)
    screen.blit(button_text, button_text_rect)
    return False

def game_over_screen(score):
    screen.fill(black)
    game_over_text = big_font.render("Game Over", True, red)
    game_over_rect = game_over_text.get_rect(center=(350, 250))
    screen.blit(game_over_text, game_over_rect)

    score_text = font.render(f"Your Score: {score}", True, white)
    score_rect = score_text.get_rect(center=(350, 320))
    screen.blit(score_text, score_rect)

    button_rect = pygame.Rect(300, 400, 200, 50)
    if draw_button("Play Again", button_rect, gray, blue):
        return True
    pygame.display.update()
    return False

def main():
    (snake_head_x, snake_head_y, snake_head_width, snake_head_height, snake_head_speed, snake_head_direction,
     apple_x, apple_y, apple_width, apple_height, score, snake_body, text, text_rect) = reset_game()

    fps = 60
    clock = pygame.time.Clock()
    running = True
    game_over = False

    while running:
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    game_over = True

            screen.fill(green)

            # Draw grid
            for x in range(0, 700, box_size):
                pygame.draw.line(screen, black, (x, 0), (x, 700))
            for y in range(0, 700, box_size):
                pygame.draw.line(screen, black, (0, y), (700, y))

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

            if snake_head_y >= 700 or snake_head_y < 0 or snake_head_x >= 700 or snake_head_x < 0:
                game_over = True

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
                apple_x = random.randint(0, (700 - box_size) // box_size) * box_size + (box_size - apple_width) // 2
                apple_y = random.randint(0, (700 - box_size) // box_size) * box_size + (box_size - apple_height) // 2
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

        # Game Over screen loop
        while game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    game_over = False

            if game_over_screen(score):
                # Reset game state to restart
                (snake_head_x, snake_head_y, snake_head_width, snake_head_height, snake_head_speed, snake_head_direction,
                 apple_x, apple_y, apple_width, apple_height, score, snake_body, text, text_rect) = reset_game()
                game_over = False

            clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
