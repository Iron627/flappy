import random
import copy
import pygame
import neat

WIDTH = 800
HEIGHT = 600
best_play = 0
draw = 1
FPS = 60
BLACK = (0, 0, 0)
GREEN = (0, 220, 0)
WHITE = (230, 230, 230)
GRAY = (80, 80, 80)
DARK_GRAY = (40, 40, 40)
POPULATION_SIZE = 100


class Bird:
    def __init__(self):
        self.x = 160
        self.size = 32
        self.color = (
            random.randint(60, 255),
            random.randint(60, 255),
            random.randint(60, 255),
        )
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
        pygame.draw.rect(screen, self.color, self.rect())


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
        self.population = neat.Population(
            Bird,
            POPULATION_SIZE,
            genome_saver=neat.save_best_genome,
        )
        self.birds = self.population.agents
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 48)
        self.button_font = pygame.font.Font(None, 32)
        self.best_score = 0
        self.manual_save_status = "No completed generation yet"
        self.manual_save_button = pygame.Rect(WIDTH - 260, 20, 230, 48)
        self.training_state = None
        self.initialized = False
        self.reset()

    def save_training_state(self):
        self.training_state = {
            "pipes": copy.deepcopy(self.pipes),
            "score": self.score,
            "dead": self.dead,
            "best_score": self.best_score,
            "population": copy.deepcopy(self.population),
            "manual_save_status": self.manual_save_status,
        }

    def restore_training_state(self):
        if self.training_state is None:
            self.birds = []
            self.reset(record_completed=False)
            return

        self.pipes = self.training_state["pipes"]
        self.score = self.training_state["score"]
        self.dead = self.training_state["dead"]
        self.best_score = self.training_state["best_score"]
        self.population = self.training_state["population"]
        self.birds = self.population.agents
        self.manual_save_status = self.training_state["manual_save_status"]
        self.training_state = None

    def toggle_best_play(self):
        global best_play
        if best_play:
            best_play = False
            self.restore_training_state()
            return

        self.save_training_state()
        best_play = True
        self.reset(record_completed=False)

    def reset(self, record_completed=True):
        if best_play:
            self.pipes = [Pipe(WIDTH + i * 260) for i in range(3)]
            self.score = 0
            self.dead = False
            self.birds = [Bird()]
            self.load_best_genome(self.birds[0])
            return

        if not self.initialized:
            self.pipes = [Pipe(WIDTH + i * 260) for i in range(3)]
            self.score = 0
            self.dead = False
            self.initialized = True
            return

        score_achieved = getattr(self, "score", 0)
        self.pipes = [Pipe(WIDTH + i * 260) for i in range(3)]
        self.score = 0
        self.dead = False
        self.population.evolve(score_achieved, record_completed=record_completed)
        self.birds = self.population.agents
        self.manual_save_status = (
            f"Ready: gen {self.population.latest_completed_generation}, score {self.population.latest_completed_score}"
            if self.population.latest_completed_top_genome is not None and record_completed else self.manual_save_status
        )

    def save_manual_genome(self):
        if self.population.latest_completed_top_genome is None:
            self.manual_save_status = "No completed generation yet"
            return

        saved = neat.save_manual_genome(
            self.population.latest_completed_top_genome,
            self.population.latest_completed_generation,
            self.population.latest_completed_score,
        )
        if not saved:
            self.manual_save_status = f"Already saved gen {self.population.latest_completed_generation}"
            return

        self.manual_save_status = f"Saved gen {self.population.latest_completed_generation}"

    def load_best_genome(self, bird):
        bird.neuron.genome = neat.load_best_genome()

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
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not best_play:
                if self.manual_save_button.collidepoint(event.pos):
                    self.save_manual_genome()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit()
                if event.key == pygame.K_c:
                    FPS = 0 if FPS else 60
                if event.key == pygame.K_d:
                    draw = not draw
                if event.key == pygame.K_b:
                    self.toggle_best_play()
                
                

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

        self.population.update_best_fitness()
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
        labels = [] if best_play else [
            f"Alive: {alive_count} Generation: {self.population.generation}",
            f"Best Score: {self.best_score}",
            f"Average Score: {self.population.average_score():.2f}",
            f"Best Fitness: {self.population.best_fitness}",
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
            if not best_play:
                self.draw_manual_save_button()

             
        else:
            self.screen.fill(BLACK)
            score = self.font.render(str(self.score), True, WHITE)
            self.screen.blit(score, (WIDTH // 2, HEIGHT-30))

            for i, label in enumerate(labels):
                text = self.font.render(label, True, WHITE)
                self.screen.blit(text, (20, 30 + i * 45))
            if not best_play:
                self.draw_manual_save_button()

        pygame.display.flip()

    def draw_manual_save_button(self):
        button_color = GRAY if self.population.latest_completed_top_genome else DARK_GRAY
        pygame.draw.rect(self.screen, button_color, self.manual_save_button, border_radius=8)
        pygame.draw.rect(self.screen, WHITE, self.manual_save_button, 2, border_radius=8)

        label = self.button_font.render("Save Top Bird", True, WHITE)
        label_rect = label.get_rect(center=self.manual_save_button.center)
        self.screen.blit(label, label_rect)

        status = self.button_font.render(self.manual_save_status, True, WHITE)
        self.screen.blit(status, (self.manual_save_button.x, self.manual_save_button.bottom + 8))

    def quit(self):
        pygame.quit()
        raise SystemExit


Game().run()
