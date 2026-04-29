import random
import numpy as np


class NEAT_Lite:
    def __init__(self):
        self.genome = {
    "weight": np.array([[random.uniform(-1, 1) for _ in range(3)] for _ in range(5)]),
    "bias": np.array([random.uniform(-1, 1) for _ in range(3)]),
    "weight1": np.array([random.uniform(-1, 1) for _ in range(3)]),
    "bias1": random.uniform(-1, 1)
}
        pass
    def forward(self, inputs):
        inputs = np.array(inputs)
        hidden = inputs@self.genome["weight"] + self.genome["bias"]
        hidden = np.tanh(hidden)
        layer_output = hidden@self.genome["weight1"] + self.genome["bias1"]
        return np.tanh(layer_output) > 0

    def mutate(self):
        for i in range(self.genome["weight"].shape[0]):
            for j in range(self.genome["weight"].shape[1]):
                if random.random() < 0.1:
                    self.genome["weight"][i][j] += random.uniform(-0.5, 0.5)

        for i in range(len(self.genome["bias"])):
            if random.random() < 0.1:
                self.genome["bias"][i] += random.uniform(-0.5, 0.5)

        for i in range(len(self.genome["weight1"])):
            if random.random() < 0.1:
                self.genome["weight1"][i] += random.uniform(-0.5, 0.5)

        if random.random() < 0.1:
            self.genome["bias1"] += random.uniform(-0.5, 0.5)

        
class NEAT:
    def __init__(self):
        self.nodes = {
            0 : "input",
            1 : "input",
            2: "input",
            3: "input",
            4: "input",
            5:"bias",
            10**9: "output",
        }
        self.next_node_id = 6
        self.connections = [
            {"in": 0, "out": 10**9, "weight": random.uniform(-1, 1), "enabled": True},
            {"in": 1, "out": 10**9, "weight": random.uniform(-1, 1), "enabled": True},
            {"in": 2, "out": 10**9, "weight": random.uniform(-1, 1), "enabled": True},
            {"in": 3, "out": 10**9, "weight": random.uniform(-1, 1), "enabled": True},
            {"in": 4, "out": 10**9, "weight": random.uniform(-1, 1), "enabled": True},
            {"in": 5, "out": 10**9, "weight": random.uniform(-1, 1), "enabled": True}
            
            ]
    def mutate_connection(self):
        ins = list(self.nodes.keys())
        outs = [_ for _ in self.nodes.keys() if self.nodes[_] != "input"]
        a = random.choice(ins)
        b = random.choice(outs)
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
        values[5] = 1
        for node in sorted(self.nodes.keys()):
            if self.nodes[node] in ["bias","input"]:
                continue
            
            tot_signals = 0 
            for conn in self.connections:
                if conn["out"] == node and conn["enabled"]:
                    tot_signals += values.get(conn["in"],0) * conn["weight"]
            values[node] = np.tanh(tot_signals)
        return values[10**9] > 0