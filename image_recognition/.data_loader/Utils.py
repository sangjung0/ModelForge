import csv
from tqdm import tqdm
from pathlib import Path

from data_loader.Env import *


class Utils:
  @staticmethod
  def get_max_length_of_path(path:Path):
    max_length = 0
    with open(path, "r", buffering=Env.READ_BUFFER_SIZE.value) as f:
      reader = csv.reader(f)
      for row in tqdm(reader, desc=f"🟩 Check max length of path"):
        max_length = max(max_length, len(row[0].encode("utf-8")))
    return max_length

  @staticmethod
  def get_max_index(path:Path):
    with open(path, "r", buffering=Env.READ_BUFFER_SIZE.value) as f:
        max_index = sum(1 for _ in tqdm(f, desc=f"🟩 Check max index"))
    return max_index