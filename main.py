import random
import copy
import pygame
import neat
import numpy as np

WIDTH = 800
HEIGHT = 600
best_play = 1
FPS = 60 if best_play else 0
BLACK = (0, 0, 0)
GREEN = (0, 220, 0)
WHITE = (230, 230, 230)
POPULATION_SIZE = 100


class Bird:
    def __init__(self):
        self.x = 160
        self.size = 32
        self.neuron = neat.NEAT_Lite()
        self.reset()

    def reset(self):
        self.y = HEIGHT // 2
        self.vel = 0
        self.alive = True
        self.fitness = 0

    def rect(self):
        return pygame.Rect(self.x, self.y, self.size, self.size)

    def flap(self):
        self.vel = -8

    def update(self):
        self.vel += 0.45
        self.y += self.vel

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.rect())


class Pipe:
    def __init__(self, x):
        self.x = x
        self.w = 70
        self.gap = 170
        self.gap_y = random.randint(140, HEIGHT - 140)
        self.passed = False

    def top_rect(self):
        return pygame.Rect(self.x, 0, self.w, self.gap_y - self.gap // 2)

    def bottom_rect(self):
        y = self.gap_y + self.gap // 2
        return pygame.Rect(self.x, y, self.w, HEIGHT - y)

    def update(self):
        self.x -= 4

    def hit(self, bird):
        return bird.rect().colliderect(self.top_rect()) or bird.rect().colliderect(self.bottom_rect())

    def draw(self, screen):
        pygame.draw.rect(screen, GREEN, self.top_rect())
        pygame.draw.rect(screen, GREEN, self.bottom_rect())


class Game:
    def __init__(self):
        pygame.init()
        self.birds = []
        self.generation = 1 
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        self.reset()

    def reset(self):
        if best_play:
            self.pipes = [Pipe(WIDTH + i * 260) for i in range(3)]
            self.score = 0
            self.dead = False
            self.best_fitness = 0
            self.birds = [Bird()]
            self.load_best_genome(self.birds[0])
            return

        best_birds = sorted(self.birds, key=lambda g: g.fitness, reverse=True)
        parent = copy.deepcopy(best_birds[0]) if best_birds else None
        self.pipes = [Pipe(WIDTH + i * 260) for i in range(3)]
        self.score = 0
        self.dead = False
        self.best_fitness = 0
        self.generation += 1
        if parent:
            with open("best_genome.txt", "w") as f:
                f.write(str(parent.neuron.genome))
        for _ in range(POPULATION_SIZE):
            bird = Bird()
            if parent:
                bird.neuron.genome = copy.deepcopy(parent.neuron.genome)
                bird.neuron.mutate()
            self.birds.append(bird)
        else:
            self.birds = [Bird() for _ in range(POPULATION_SIZE)]

    def load_best_genome(self, bird):
        with open("best_genome.txt") as f:
            bird.neuron.genome = eval(
                f.read(),
                {"__builtins__": {}},
                {"array": np.array, "np": np},
            )

    def run(self):
        while True:
            self.events()
            if not self.dead:
                self.update()
            self.draw()
            if FPS > 0:
                self.clock.tick(FPS)
            

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit()
                
                

    def update(self):
        alive_birds = [bird for bird in self.birds if bird.alive]
        

        for bird in alive_birds:
            pipe = self.next_pipe_for(bird)
            inputs = [
                bird.y / HEIGHT,
                bird.vel / 10,
                (pipe.x - bird.x) / WIDTH,
                (pipe.gap_y - bird.y) / HEIGHT,
                pipe.gap_y / HEIGHT,
            ]
            if bird.neuron.forward(inputs):
                bird.flap()
            bird.update()
            bird.fitness += 1

        for pipe in self.pipes:
            pipe.update()

            for bird in alive_birds:
                if bird.alive and pipe.hit(bird):
                    bird.alive = False

            if not pipe.passed and pipe.x + pipe.w < self.birds[0].x:
                pipe.passed = True
                self.score += 1
                for bird in self.birds:
                    if bird.alive:
                        bird.fitness += 10

        if self.pipes[0].x + self.pipes[0].w < 0:
            self.pipes.pop(0)
            self.pipes.append(Pipe(self.pipes[-1].x + 260))

        for bird in alive_birds:
            if bird.alive and (bird.y < 0 or bird.y + bird.size > HEIGHT):
                bird.alive = False

        self.best_fitness = max(bird.fitness for bird in self.birds)
        self.dead = not any(bird.alive for bird in self.birds)
    def next_pipe_for(self, bird):
        for pipe in self.pipes:
            if pipe.x + pipe.w >= bird.x:
                return pipe
        return self.pipes[0]

    def draw(self):
        self.screen.fill(BLACK)
        for pipe in self.pipes:
            pipe.draw(self.screen)
        for bird in self.birds:
            if bird.alive:
                bird.draw(self.screen)

        score = self.font.render(str(self.score), True, WHITE)
        self.screen.blit(score, (WIDTH // 2, HEIGHT-30))

        alive_count = sum(bird.alive for bird in self.birds)
        label = "Best Genome Play" if best_play else f"Alive: {alive_count} Generation: {self.generation}"
        alive = self.font.render(label, True, WHITE)
        self.screen.blit(alive, (20, 30))

        if self.dead:
            self.reset()

        pygame.display.flip()

    def quit(self):
        pygame.quit()
        raise SystemExit


Game().run()
