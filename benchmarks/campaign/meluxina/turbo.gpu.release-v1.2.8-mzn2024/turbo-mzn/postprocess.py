from pathlib import Path
import ast
import json
import os
import re
import sys

import minizinc
from ruamel.yaml import YAML

yaml = YAML(typ="safe")
yaml.register_class(minizinc.types.ConstrEnum)
yaml.register_class(minizinc.types.AnonEnum)
yaml.default_flow_style = False


if os.environ.get("MZN_DEBUG", "OFF") == "ON":
  import logging
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


def default_statistics(uid):
  return {
    "configuration": uid,
    "status": str(minizinc.result.Status.UNKNOWN)
  }


def parse_solution_value(value):
  value = value.strip()
  if value == "true":
    return True
  if value == "false":
    return False
  if re.fullmatch(r"-?\d+", value):
    return int(value)
  if re.fullmatch(r"-?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?", value):
    return float(value)
  if value.startswith("[") or value.startswith("{") or value.startswith("("):
    normalized = re.sub(r"\btrue\b", "True", value)
    normalized = re.sub(r"\bfalse\b", "False", normalized)
    try:
      return ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
      return value
  if len(value) >= 2 and value[0] == "\"" and value[-1] == "\"":
    return value[1:-1]
  return value


def parse_raw_solution(raw_output):
  if not isinstance(raw_output, str):
    return None
  solution = {}
  for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?);", raw_output, re.DOTALL):
    name = match.group(1)
    value = match.group(2)
    solution[name] = parse_solution_value(value)
  return solution or None


def extract_solution_payload(output):
  payload = output.get("output", {})
  if not isinstance(payload, dict):
    return {"_raw": str(payload)}
  if "json" in payload:
    return payload["json"]
  raw_output = payload.get("raw", payload.get("default"))
  parsed = parse_raw_solution(raw_output)
  if parsed is not None:
    return parsed
  if raw_output is not None:
    return {"_raw": raw_output}
  return {}


if __name__ == "__main__":
  output_dir = sys.argv[1]
  if output_dir[-1] == "/":
    output_dir = output_dir[:-1]
  input_file = Path(sys.argv[2])
  uid = input_file.stem

  sol_filename = Path(output_dir + "/" + uid + "_sol.yml")
  stats_filename = Path(output_dir + "/" + uid + "_stats.yml")

  statistics = default_statistics(uid)
  solutions = []
  unknowns = ""
  errors = ""
  with open(input_file) as input_json:
    for line in input_json:
      try:
        output = json.loads(line)
      except json.JSONDecodeError:
        unknowns += line
        continue
      # We keep successive experiments in the JSON file, even when they fail.
      # For the statistics, we are only interested by the latest experiment, starting with {"type": "lattice-land", "lattice-land": "start"}.
      if output["type"] == "lattice-land":
        if output["lattice-land"] == "start":
          statistics = default_statistics(uid)
          unknowns = ""
          errors = ""
      elif output["type"] == "comment":
        pass
      elif output["type"] == "statistics":
        statistics.update(output["statistics"])
      elif output["type"] == "status":
        statistics["status"] = output["status"]
      elif output["type"] == "solution":
        sol = statistics.copy()
        sol["status"] = str(minizinc.result.Status.SATISFIED)
        if statistics["status"] == str(minizinc.result.Status.UNKNOWN):
          statistics["status"] = str(minizinc.result.Status.SATISFIED)
        sol["solution"] = extract_solution_payload(output)
        if "_objective" in sol["solution"] and "objective" not in sol["solution"]:
          sol["solution"]["objective"] = sol["solution"]["_objective"]
        if "_objective" in sol["solution"]:
          del sol["solution"]["_objective"]
        if "time" in output:
          sol["time"] = float(output["time"]) / 1000.0
        solutions.append(sol)
      elif output["type"] == "error":
        statistics["status"] = str(minizinc.result.Status.ERROR)
        errors += line
      else:
        unknowns += line

  # Ignore errors if it could find a solution.
  if statistics["status"] == str(minizinc.result.Status.UNKNOWN) or statistics["status"] == str(minizinc.result.Status.ERROR):
    if errors != "" or "Exception" in unknowns:
      print(sys.argv[2], file=sys.stderr)

  if solutions != []:
    with open(sol_filename, "w") as file:
      yaml.dump(solutions, file)
  with open(stats_filename, "w") as file:
    yaml.dump(statistics, file)

