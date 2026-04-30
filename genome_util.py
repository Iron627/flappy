import json
from pathlib import Path


BEST_GENOME_DIR = Path("best_genomes")
BEST_GENOME_PREFIX = "best_genome_"


def save_best_genome(genome, score):
    BEST_GENOME_DIR.mkdir(exist_ok=True)
    genome_path = BEST_GENOME_DIR / f"{BEST_GENOME_PREFIX}{score}.json"
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


def load_genome_file(genome_path):
    with Path(genome_path).open() as f:
        return json.load(f)
