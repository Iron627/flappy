import random
import copy
from pathlib import Path

import pygame
import neat
import genome_util

WIDTH = 800
HEIGHT = 600
best_play = 0
draw = 1
FPS = 60
BLACK = (0, 0, 0)
GREEN = (0, 220, 0)
WHITE = (230, 230, 230)
GRAY = (80, 80, 80)
POPULATION_SIZE = 100


class Bird:
    def __init__(self):
        self.x = 160
        self.size = 32
        self.neuron = neat.NEAT()
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
        self.generation = 0
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        self.button_font = pygame.font.Font(None, 32)
        self.best_score = 0
        self.generation_scores = []
        self.loaded_genome_path = None
        self.load_genome_status = "Using best saved genome"
        self.load_genome_button = pygame.Rect(WIDTH - 280, 20, 250, 48)
        self.reset()

    def reset(self, record_completed=True):
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
        score_achieved = getattr(self, "score", 0)
        self.pipes = [Pipe(WIDTH + i * 260) for i in range(3)]
        self.score = 0
        self.dead = False
        self.best_fitness = 0
        self.generation += 1
        if parent:
            if record_completed:
                self.generation_scores.append(score_achieved)
            self.birds = []
            for _ in range(POPULATION_SIZE):
                bird = Bird()
                if parent:
                    bird.neuron.genome = copy.deepcopy(parent.neuron.genome)
                    bird.neuron.mutate()
                self.birds.append(bird)
            if record_completed:
                genome_util.save_best_genome(parent.neuron.genome, score_achieved)
        else:
            self.birds = [Bird() for _ in range(POPULATION_SIZE)]

    def average_generation_score(self):
        if not self.generation_scores:
            return 0
        return sum(self.generation_scores) / len(self.generation_scores)

    def load_best_genome(self, bird):
        if self.loaded_genome_path:
            bird.neuron.genome = genome_util.load_genome_file(self.loaded_genome_path)
        else:
            bird.neuron.genome = genome_util.load_best_genome()

    def choose_genome_file(self):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            genome_path = filedialog.askopenfilename(
                title="Load genome JSON",
                filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
            )
            root.destroy()
        except Exception as error:
            self.load_genome_status = f"Could not open file picker: {error}"
            return

        if not genome_path:
            return

        try:
            genome_util.load_genome_file(genome_path)
        except (OSError, ValueError) as error:
            self.load_genome_status = f"Could not load JSON: {error}"
            return

        self.loaded_genome_path = genome_path
        self.load_genome_status = f"Loaded: {Path(genome_path).name}"
        self.reset(record_completed=False)

    def run(self):
        while True:
            self.events()
            if not self.dead:
                self.update()
            self.draw()
            if FPS > 0:
                self.clock.tick(FPS)
            

    def events(self):
        global FPS, draw, best_play
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if best_play and self.load_genome_button.collidepoint(event.pos):
                    self.choose_genome_file()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit()
                if event.key == pygame.K_c:
                    FPS = 0 if FPS else 60
                if event.key == pygame.K_d:
                    draw = not draw
                if event.key == pygame.K_b:
                    best_play = not best_play
                    self.reset(record_completed=False)
                
                

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
                self.best_score = max(self.best_score, self.score)
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
        if self.dead:
                self.reset()
    def next_pipe_for(self, bird):
        for pipe in self.pipes:
            if pipe.x + pipe.w >= bird.x:
                return pipe
        return self.pipes[0]

    def draw(self):
        alive_count = sum(bird.alive for bird in self.birds)
        if best_play:
            labels = ["Best Genome Play"]
        else:
            labels = [
                f"Alive: {alive_count} Generation: {self.generation}",
                f"Best Score: {self.best_score}",
                f"Average Score: {self.average_generation_score():.2f}",
                f"Best Fitness: {self.best_fitness}",
            ]

        if draw:
            self.screen.fill(BLACK)
            for pipe in self.pipes:
                pipe.draw(self.screen)
            for bird in self.birds:
                if bird.alive:
                    bird.draw(self.screen)

            score = self.font.render(str(self.score), True, WHITE)
            self.screen.blit(score, (WIDTH // 2, HEIGHT-30))

            for i, label in enumerate(labels):
                text = self.font.render(label, True, WHITE)
                self.screen.blit(text, (20, 30 + i * 45))
            self.draw_load_genome_button()

             
        else:
            self.screen.fill(BLACK)
            score = self.font.render(str(self.score), True, WHITE)
            self.screen.blit(score, (WIDTH // 2, HEIGHT-30))

            for i, label in enumerate(labels):
                text = self.font.render(label, True, WHITE)
                self.screen.blit(text, (20, 30 + i * 45))
            self.draw_load_genome_button()

        pygame.display.flip()

    def draw_load_genome_button(self):
        if not best_play:
            return

        pygame.draw.rect(self.screen, GRAY, self.load_genome_button, border_radius=8)
        pygame.draw.rect(self.screen, WHITE, self.load_genome_button, 2, border_radius=8)

        label = self.button_font.render("Load Genome JSON", True, WHITE)
        label_rect = label.get_rect(center=self.load_genome_button.center)
        self.screen.blit(label, label_rect)

        status = self.button_font.render(self.load_genome_status, True, WHITE)
        self.screen.blit(status, (self.load_genome_button.x, self.load_genome_button.bottom + 8))

    def quit(self):
        pygame.quit()
        raise SystemExit


Game().run()
