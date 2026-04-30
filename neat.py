import copy
import json
import random
from pathlib import Path

import numpy as np


BEST_GENOME_DIR = Path("best_genomes")
MANUAL_SAVE_DIR = BEST_GENOME_DIR / "manual_saves"
BEST_GENOME_PREFIX = "best_genome_"


def save_best_genome(genome, score):
    BEST_GENOME_DIR.mkdir(exist_ok=True)
    genome_path = BEST_GENOME_DIR / f"{BEST_GENOME_PREFIX}{score}.json"
    if genome_path.exists():
        return False

    with genome_path.open("w") as f:
        json.dump(genome, f)
    return True


def save_manual_genome(genome, generation, score):
    MANUAL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    genome_path = MANUAL_SAVE_DIR / f"manual_best_genome_gen_{generation}_score_{score}.json"
    if genome_path.exists():
        return False

    with genome_path.open("w") as f:
        json.dump(genome, f)
    return True


def best_genome_files():
    if not BEST_GENOME_DIR.exists():
        return []

    genome_files = []
    for genome_path in BEST_GENOME_DIR.glob(f"{BEST_GENOME_PREFIX}*.json"):
        score_text = genome_path.stem.removeprefix(BEST_GENOME_PREFIX)
        if score_text.isdigit():
            genome_files.append((int(score_text), genome_path))
    return genome_files


def load_best_genome():
    genome_files = best_genome_files()
    if genome_files:
        _, genome_path = max(genome_files, key=lambda genome_file: genome_file[0])
    else:
        genome_path = Path("best_genome.json")

    with genome_path.open() as f:
        return json.load(f)



        
class NEAT:
    def __init__(self, inputs, outputs):
        self.num_inputs = inputs
        self.num_outputs = outputs
        self.nodes = {}
        
        for i in range(inputs):
            self.nodes[i] = "input"
        
        self.bias_id = inputs
        self.nodes[self.bias_id] = "bias"
        

        self.output_ids = list(range(10**9, 10**9 - outputs, -1))
        for output_id in self.output_ids:
            self.nodes[output_id] = "output"
        
        self.next_node_id = inputs + 1  
        
        self.connections = []
        for input_id in range(inputs):
            for output_id in self.output_ids:
                self.connections.append({
                    "in": input_id,
                    "out": output_id,
                    "weight": random.uniform(-1, 1),
                    "enabled": True
                })
        
        for output_id in self.output_ids:
            self.connections.append({
                "in": self.bias_id,
                "out": output_id,
                "weight": random.uniform(-1, 1),
                "enabled": True
            })
    def check_cycle(self, a, b):
        graph = {node: [] for node in self.nodes}
        for conn in self.connections:
            if conn["enabled"]:
                x = conn["in"]
                y = conn["out"]
                graph[x].append(y)
        visited = set()
        def dfs(node):
            if node == b:
                return True
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
            visited.remove(node)
            return False
        return dfs(b)
    def topological_sort(self):
        n_ins = {node: 0 for node in self.nodes.keys()}
        graph = {node: [] for node in self.nodes}
        for conn in self.connections:
            if conn["enabled"]:
                a = conn["in"]
                b = conn["out"]
                graph[a].append(b)
                n_ins[b] += 1
            else:
                continue
        q = [node for node in self.nodes if n_ins[node] == 0]
        sorted = []
        while q:
            node = q.pop(0)
            sorted.append(node)
            for _ in graph[node]:
                n_ins[_] -=1
                if n_ins[_] == 0:
                    q.append(_)
        return sorted
    def mutate_connection(self):
        ins = list(self.nodes.keys())
        outs = [_ for _ in self.nodes.keys() if self.nodes[_] not in ["input","bias"]]
        a = random.choice(ins)
        b = random.choice(outs)
        if self.check_cycle(a, b):
            return
        if a == b:
            return
        if a > b:
            return
        for conn in self.connections:
            if conn["in"] == a and conn["out"] == b:
                return
        self.connections.append({"in": a, "out": b, "weight": random.uniform(-1, 1), "enabled": True})
    def mutate_node(self):
        enabled_connections = [conn for conn in self.connections if conn["enabled"]]
        if not enabled_connections:
            return
        conn = random.choice(enabled_connections)
        conn["enabled"] = False
        new_node = self.next_node_id
        self.next_node_id += 1
        self.nodes[new_node] = "hidden"
        new_conn_in = {
            "in": conn["in"],
            "out": new_node,
            "weight": 1.0,
            "enabled": True
        }
        new_conn_out = {
            "in": new_node,
            "out": conn["out"],
            "weight": conn["weight"],
            "enabled": True
        }
        self.connections.append(new_conn_in)
        self.connections.append(new_conn_out)
    def mutate(self):
        for c in self.connections:
            if random.random() < 0.1:
                c["weight"] += random.uniform(-0.5, 0.5)

        if random.random() < 0.05:
            self.mutate_connection()

        if random.random() < 0.03:
            self.mutate_node()

    def forward(self, inputs):
        values = {}

        for i, value in enumerate(inputs):
            values[i] = value
        values[self.bias_id] = 1
        for node in self.topological_sort():
            if self.nodes[node] in ["bias","input"]:
                continue
            
            tot_signals = 0 
            for conn in self.connections:
                if conn["out"] == node and conn["enabled"]:
                    tot_signals += values.get(conn["in"],0) * conn["weight"]
            values[node] = np.tanh(tot_signals)
        
        # Return True if first output > 0 (for backward compatibility with single output)
        if self.output_ids:
            return values[self.output_ids[0]] > 0
        return False
    @property
    def genome(self):
        return {
            "nodes": self.nodes,
            "connections": self.connections,
            "next_node_id": self.next_node_id
        }

    @genome.setter
    def genome(self, data):
        self.nodes = {int(k): v for k, v in data["nodes"].items()}
        self.connections = data["connections"]
        self.next_node_id = data["next_node_id"]


class Population:
    def __init__(
        self,
        agent_factory,
        size,
        brain_attr="neuron",
        fitness_attr="fitness",
        genome_saver=None,
    ):
        self.agent_factory = agent_factory
        self.size = size
        self.brain_attr = brain_attr
        self.fitness_attr = fitness_attr
        self.genome_saver = genome_saver
        self.agents = []
        self.generation = 0
        self.best_fitness = 0
        self.generation_scores = []
        self.latest_completed_top_genome = None
        self.latest_completed_generation = None
        self.latest_completed_score = None
        self.create_initial_generation()

    def _brain(self, agent):
        return getattr(agent, self.brain_attr)

    def _fitness(self, agent):
        return getattr(agent, self.fitness_attr, 0)

    def top_agent(self):
        return max(self.agents, key=self._fitness, default=None)

    def update_best_fitness(self):
        self.best_fitness = max((self._fitness(agent) for agent in self.agents), default=0)
        return self.best_fitness

    def average_score(self):
        if not self.generation_scores:
            return 0
        return sum(self.generation_scores) / len(self.generation_scores)

    def create_initial_generation(self):
        self.generation = 1
        self.agents = [self.agent_factory() for _ in range(self.size)]
        self.update_best_fitness()

    def evolve(self, score=0, record_completed=True, save_best=None):
        if save_best is None:
            save_best = record_completed

        parent = self.top_agent()
        if not parent:
            self.create_initial_generation()
            return

        parent_genome = copy.deepcopy(self._brain(parent).genome)
        if record_completed:
            completed_generation = self.generation
            self.generation_scores.append(score)
            self.latest_completed_top_genome = copy.deepcopy(parent_genome)
            self.latest_completed_generation = completed_generation
            self.latest_completed_score = score
        if save_best and self.genome_saver:
            self.genome_saver(parent_genome, score)

        self.generation += 1
        new_agents = []
        for _ in range(self.size):
            agent = self.agent_factory()
            getattr(agent, self.brain_attr).genome = copy.deepcopy(parent_genome)
            getattr(agent, self.brain_attr).mutate()
            new_agents.append(agent)
        self.agents = new_agents
        self.update_best_fitness()
