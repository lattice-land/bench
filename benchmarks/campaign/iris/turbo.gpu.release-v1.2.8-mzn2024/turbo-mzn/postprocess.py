from sys import stdin
from pathlib import Path
import sys
import os
import minizinc
import json
from ruamel.yaml import YAML

yaml = YAML(typ="safe")
yaml.register_class(minizinc.types.ConstrEnum)
yaml.register_class(minizinc.types.AnonEnum)
yaml.default_flow_style = False

if os.environ.get("MZN_DEBUG", "OFF") == "ON":
  import logging
  logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

if __name__ == "__main__":
  output_dir = sys.argv[1]
  if(output_dir[-1] == "/"):
    output_dir = output_dir[:-1]
  input_file = Path(sys.argv[2])
  uid = input_file.stem

  sol_filename = Path(output_dir + "/" + uid + "_sol.yml")
  stats_filename = Path(output_dir + "/" + uid + "_stats.yml")

  statistics = {
    "configuration": uid,
    "status": str(minizinc.result.Status.UNKNOWN)
  }
  solutions = []
  unknowns = ""
  errors = ""
  with open(input_file) as input_json:
    for line in input_json:
      output = ""
      try:
        output = json.loads(line)
      except json.JSONDecodeError:
        unknowns.append(line)
        continue
      # We keep successive experiments in the JSON file, even when they fail.
      # For the statistics, we are only interested by the latest experiment, starting with {"type": "lattice-land", "lattice-land": "start"}.
      if(output["type"] == "lattice-land"):
        if(output["lattice-land"] == "start"):
          statistics = {}
          unknowns = []
          errors = []
      elif(output["type"] == "comment"):
        pass
      elif(output["type"] == "statistics"):
        statistics.update(output["statistics"])
      elif(output["type"] == "status"):
        statistics["status"] = output["status"]
      elif(output["type"] == "solution"):
        sol = statistics.copy()
        sol["status"] = str(minizinc.result.Status.SATISFIED)
        if(statistics["status"] == str(minizinc.result.Status.UNKNOWN)):
          statistics["status"] = str(minizinc.result.Status.SATISFIED)
        sol["solution"] = output["output"]["json"]
        if("_objective" in sol["solution"] and "objective" not in sol["solution"]):
          sol["solution"]["objective"] = sol["solution"]["_objective"]
        if("_objective" in sol["solution"]):
          del sol["solution"]["_objective"]
        sol["time"] = float(output["time"]) / 1000.0
        solutions.append(sol)
      elif(output["type"] == "error"):
        errors += line
      else:
        unknowns += line

  # Ignore errors if it could find a solution.
  if statistics["status"] == str(minizinc.result.Status.UNKNOWN) or statistics["status"] == str(minizinc.result.Status.ERROR):
    if errors != [] or "Exception" in unknowns:
      print(sys.argv[2], file=sys.stderr)

  if solutions != []:
    with open(sol_filename, "w") as file:
      yaml.dump(solutions, file)
  with open(stats_filename, "w") as file:
    yaml.dump(statistics, file)
