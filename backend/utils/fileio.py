import csv
import json
from pathlib import Path


def read_csv(filepath: Path) -> list[dict[str, str]]:
    data = []
    with open(filepath, "r") as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return data


def save_to_csv(filepath: Path, field_names: list, data: list):
    with open(filepath, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def read_json(filepath: Path) -> dict:
    with open(filepath, "r") as file:
        data = json.load(file)
    return data


def save_to_json(data: dict, filepath: Path):
    with open(filepath, mode="w") as file:
        json.dump(data, file, indent=4)


def save_to_file(data: str, filepath: Path, mode: str = "w"):
    with open(filepath, mode=mode) as file:
        file.write(data + "\n")
