import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import seaborn as sns
import scipy.cluster.hierarchy as sch
from pathlib import Path
from packaging import version
from matplotlib.colors import LinearSegmentedColormap
import textwrap

def determine_version(solver_info):
  if 'choco' in solver_info:
    return "4.10.15"
  elif 'ortools' in solver_info:
    return "9.9"
  else:
    return "unknown"

# A tentative to have unique experiment names.
def make_uid(config, arch, mzn_solver, version, machine, timeout_ms):
  uid = mzn_solver + "_" + str(version) + '_' + machine
  if str(timeout_ms) == "inf":
    uid += "_notimeout"
  elif(math.isnan(timeout_ms)):
    uid += "_notimeout"
  elif (int(timeout_ms) % 1000) == 0:
    uid += "_" + str(int(int(timeout_ms)/1000)) + "s"
  else:
    uid += "_" + str(int(timeout_ms)) + "ms"
  if arch != "":
    uid += "_" + arch.lower()
  if 'java11' in config:
    uid += '_java11'
  if mzn_solver == 'turbo.gpu.release':
    if '_disable_simplify' in config:
      uid += '_disable_simplify'
    if '_force_ternarize' in config:
      uid += '_force_ternarize'
    if '_ipc' in config:
      uid += '_ipc'
  return uid

def make_short_uid(uid):
  components = uid.split('_')
  mzn_solver = components[0]
  if mzn_solver == 'turbo.gpu.release':
    mzn_solver = 'turbo.gpu'
  elif mzn_solver == 'turbo.cpu.release':
    mzn_solver = 'turbo.cpu'
  elif mzn_solver == 'com.google.ortools.sat':
    mzn_solver = 'ortools'
  elif mzn_solver == 'com.google.ortools.sat.noglobal':
    mzn_solver = 'ortools.noglobal'
  elif mzn_solver == 'org.choco.choco':
    mzn_solver = 'choco'
  elif mzn_solver == 'org.choco.choco.noglobal':
    mzn_solver = 'choco.noglobal'
  extra = '_'.join(components[4:]) # We remove the timeout info.
  if extra != '':
    extra = "_" + extra
  return mzn_solver + '_' + components[1] + extra

def read_experiments(experiments):
  all_xp = pd.DataFrame()
  for e in experiments:
    df = pd.read_csv(e)
    df.rename(columns={"timeout_sec": "timeout_ms"}) # due to a mistake in the naming of that column.
    if "java11" in e:
      df['configuration'] = df['configuration'].apply(lambda x: x + "_java11")
    if 'version' not in df:
      df['version'] = df['configuration'].apply(determine_version)
    else:
      df['version'] = df.apply(lambda row: determine_version(row["configuration"]) if not isinstance(row['version'], str) else row['version'], axis=1)
    if 'machine' not in df:
      df['machine'] = os.path.basename(os.path.dirname(e))
    if 'timeout_ms' not in df:
      df['timeout_ms'] = "300000"
    if df[df['status'] == 'ERROR'].shape[0] > 0:
      print(e, ': Number of erroneous rows: ', df[df['status'] == 'ERROR'].shape[0])
      print(e, df[df['status'] == 'ERROR']['data_file'])
      df = df[df['status'] != 'ERROR']
    all_xp = pd.concat([df, all_xp], ignore_index=True)
  all_xp['version'] = all_xp['version'].apply(version.parse)
  if 'memory_configuration' not in all_xp:
    all_xp['memory_configuration'] = 'RAM'
  all_xp['arch'] = all_xp['arch'].fillna("").astype(str) if 'arch' in all_xp else ""
  all_xp['propagator_mem'] = all_xp['propagator_mem'].fillna(0).astype(int) if 'propagator_mem' in all_xp else 0
  all_xp['store_mem'] = all_xp['store_mem'].fillna(0).astype(int) if 'store_mem' in all_xp else 0
  all_xp['eps_num_subproblems'] = pd.to_numeric(all_xp['eps_num_subproblems'], errors='coerce').fillna(1).astype(int) if 'eps_num_subproblems' in all_xp else 1
  all_xp['hardware'] = all_xp['machine'].apply(determine_hardware)
  all_xp['cores'] = all_xp['cores'].fillna(1).astype(int)
  # For the fixpoint, we default it to "" or "ac1" if it is a Turbo solver.
  all_xp['uid'] = all_xp.apply(lambda row: make_uid(row['configuration'], row['arch'], row['mzn_solver'], row['version'], row['machine'], row['timeout_ms']), axis=1)
  all_xp['short_uid'] = all_xp['uid'].apply(make_short_uid)
  all_xp = all_xp.copy() # to avoid a warning about fragmented frame.
  all_xp['normalized_propagator_mem'] = 0
  all_xp['normalized_store_mem'] = 0
  all_xp['problem_uid'] = all_xp.apply(lambda row: f"{row['problem']}_{Path(row['data_file']).stem}", axis=1)
  # Some models don't have a data file...
  all_xp['model_data_file'] = all_xp['model'] + ' - ' + all_xp['data_file']
  return all_xp

def intersect(df):
  # Group by 'mzn_solver' and convert the 'model_data_file' column to a set
  solver_instance = df.groupby('uid')['model_data_file'].apply(set)

  # Intersection of the solvers' instances.
  target_set = solver_instance.iloc[0]
  for i in range(0, len(solver_instance)):
    # print(f"{i} has {len(solver_instance.iloc[i])} instances.")
    # print(target_set.difference(solver_instance.iloc[i]))
    target_set = target_set.intersection(solver_instance.iloc[i])

  return df[df['model_data_file'].isin(target_set)]

def determine_hardware(machine_name):
  if machine_name == 'precision5820':
    return 'Intel Core i9-10900X 10-Core@3.7GHz;24GO DDR4;NVIDIA RTX A5000'
  elif machine_name == 'meluxina':
    return 'AMD EPYC 7452 32-Core@2.35GHz;RAM 512GO;NVIDIA A100 40GB HBM'
  elif machine_name == 'aion':
    return 'AMD EPYC ROME 7H12 64-Core@2.6GHz;RAM 256GO'
  else:
    return 'unknown'

def percent(new, old):
  if old == 0:
    return "inf"
  else:
    v = round(((new - old) / old) * 100)
    if v > 0:
      return f"+{v}%"
    elif v < 0:
      return f"{v}%"
    else:
      return "0%"

def print_table_line(m1, m2, header, key, unit):
  print(f"| {header} | {m2['avg_normalized_'+key]:.2f} | {percent(m2['avg_normalized_'+key], m1['avg_normalized_'+key])} | {m2['best_'+key]} | {m2['avg_'+key]:.2f}{unit} | {percent(m2['avg_'+key], m1['avg_'+key])} | {m2['median_'+key]:.2f}{unit} | {percent(m2['median_'+key], m1['median_'+key])} |")

def compute_normalized_value(df, row, uid1, uid2, key):
  # uid1 is the solver we are created the normalized value for, and uid2 is the solver we compare to.
  uid1_ = uid1 if row['uid'] == uid1 else uid2
  uid2_ = uid2 if row['uid'] == uid1 else uid1
  uid1 = uid1_
  uid2 = uid2_

  # Retrieve the value for each solver for the particular data_file of the row examined.
  nps1 = row[key]
  nps_vals = df[(df['uid'] == uid2) & (df['model_data_file'] == row['model_data_file'])][key].values
  assert(len(nps_vals) == 1)
  nps2 = nps_vals[0]

  if nps1 < nps2:
    return nps1 / nps2 * 100
  else:
    return 100

def normalize(df, uid1, uid2, key):
  df2 = df[df['uid'].isin([uid1, uid2])]
  df2['normalized_'+key] = df2.apply(lambda row: compute_normalized_value(df2, row, uid1, uid2, key), axis=1)
  # print(df2[['uid', 'model_data_file', 'normalized_'+key, key]])
  return df2

def plot_time_distribution(arch, df):
  hybrid_time_cols = [
    "preprocessing_time",
    "fixpoint_time",
    "search_time",
    "wait_cpu_time",
    "select_fp_functions_time",
    "transfer_cpu2gpu_time",
    "transfer_gpu2cpu_time"
  ]
  gpu_time_cols = [
    "preprocessing_time",
    "fixpoint_time",
    "search_time",
    "select_fp_functions_time"
  ]
  cpu_time_cols = [
    "preprocessing_time",
    "fixpoint_time",
    "search_time",
    "select_fp_functions_time"
  ]
  time_columns = hybrid_time_cols
  if arch == "cpu":
    time_columns = cpu_time_cols
  elif arch == "gpu":
    time_columns = gpu_time_cols

  # Remove the problems that could be solved (not unknown).
  df = df[(df['status'] != 'OPTIMAL_SOLUTION') & (df['status'] != 'UNSATISFIABLE')]

  df.sort_values(by="problem_uid", ascending=False, inplace=True)
  # Set the problem names as index
  df.set_index("problem_uid", inplace=True)

  num_row = df.shape[0]

  # Plot
  colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
  ax = df[time_columns].plot(kind='barh', stacked=True, figsize=(10, num_row / 2.5), color=colors)

  # Add labels and title
  plt.xlabel("Time (seconds)")
  plt.ylabel("Problem")
  plt.title("Time Distribution in Solver Components for Each Problem (arch = " + arch + ")")
  plt.legend(title="Time Component", bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.tight_layout()

  # Add num_blocks_done labels to the right of the bars if not 0
  for idx, (index, row) in enumerate(df.iterrows()):
      total_time = row[time_columns].sum()
      num_blocks = row.get("num_blocks_done", 0)
      if num_blocks != 0:
          ax.text(
              total_time + 0.5,  # slightly to the right of the end of the bar
              idx,               # vertical position
              str(int(num_blocks)),  # convert to string for display
              va='center', ha='left', fontsize=8, color='black'
          )

  # Show plot
  plt.show()

def analyse_tnf_per_problem(df, logy, source_vars, source_cons, target_vars, target_cons, list_problems=False):
  if list_problems:
    print(f"| Problem | Data | #Vars | #Vars (TNF) | #Constraints | #Constraints (TNF) | Preprocessing (sec) |")
    print("|----------|------|-------|-------------|--------------|--------------------|---------------------|")
  all_vars_increase = []
  all_cons_increase = []
  preprocessing_times = []
  leq_10x_increase = 0
  leq_100x_increase = 0
  for index, row in df.iterrows():
    vars_increase = row[target_vars] / row[source_vars]
    cons_increase = row[target_cons] / row[source_cons]
    if vars_increase <= 10. and cons_increase <= 10.:
      leq_10x_increase += 1
    if vars_increase <= 100. and cons_increase <= 100.:
      leq_100x_increase += 1
    all_vars_increase.append(vars_increase)
    all_cons_increase.append(cons_increase)
    preprocessing_times.append(row['preprocessing_time'])
    data_name = Path(row['data_file']).stem
    if data_name == "empty":
      data_name = Path(row['model']).stem
    if list_problems:
      print(f"| {row['problem']} | {data_name} | {row[source_vars]} | {row[target_vars]} (x{vars_increase:.2f}) | {row[source_cons]} | {row[target_cons]} (x{cons_increase:.2f}) | {row['preprocessing_time']} |")
  num_problems = df.shape[0]
  print(f"average_vars_increase={sum(all_vars_increase)/num_problems:.2f}")
  print(f"average_cons_increase={sum(all_cons_increase)/num_problems:.2f}")
  print(f"max_vars_increase={max(all_vars_increase):.2f}")
  print(f"max_cons_increase={max(all_cons_increase):.2f}")
  print(f"median_vars_increase={sorted(all_vars_increase)[num_problems//2]:.2f}")
  print(f"median_cons_increase={sorted(all_cons_increase)[num_problems//2]:.2f}")
  print(f"stddev_vars_increase={np.std(all_vars_increase, ddof=0):.2f}")
  print(f"stddev_cons_increase={np.std(all_cons_increase, ddof=0):.2f}")
  print(f"leq_10x_increase={leq_10x_increase}")
  print(f"leq_100x_increase={leq_100x_increase}")

  # Plotting the increase of variables and constraints.
  fig = plt.figure(figsize=(8, 6))
  plt.scatter(all_vars_increase, all_cons_increase, color='b', marker='x', label="MiniZinc instances")
  plt.xscale('log')
  if logy:
    plt.yscale('log')
    plt.ylabel("Constraints (log scale)")
  else:
    plt.ylabel("Constraints")
  plt.xlabel("Variables (log scale)")
  # plt.title("Increase in Constraints and Variables after Preprocessing")
  plt.grid(True, which="both", linestyle="--", linewidth=0.5)
  plt.legend()
  fig.savefig("tnf-increase.pdf")
  plt.show()

def preprocessing_time_distribution(df):
  preprocessing_times = []
  for _, row in df.iterrows():
    preprocessing_times.append(row['preprocessing_time'])
  num_problems = df.shape[0]
  print(f"average_preprocessing_time={sum(preprocessing_times)/num_problems:.2f}")
  print(f"median_preprocessing_time={sorted(preprocessing_times)[num_problems//2]:.2f}")
  print(f"stddev_preprocessing_time={np.std(preprocessing_times, ddof=0):.2f}")
  print(f"max_preprocessing_time={max(preprocessing_times):.2f}")

  # Plotting the distribution of preprocessing time
  bins = [0.01,0.1,1,10,100,1000,10000]
  if min(preprocessing_times) < bins[0] or max(preprocessing_times) > bins[-1]:
    print("WARNING: Data points outside of the histogram bins!!")
  fig = plt.figure(figsize=(8, 6))
  counts, bin_edges, patches = plt.hist(preprocessing_times, bins=bins, color='blue', edgecolor='black', alpha=0.7)
  # Annotate each bar with its frequency
  for count, left_edge, right_edge in zip(counts, bin_edges[:-1], bin_edges[1:]):
    bin_center = np.sqrt(left_edge * right_edge)  # Log-space centering
    plt.text(bin_center, count + 0.4, str(int(count)), ha='center', fontsize=10)
  plt.xscale('log')
  plt.xlabel("Preprocessing Time (seconds) [log scale]")
  plt.ylabel("Instances")
  plt.title("Log-Spaced Histogram of Preprocessing Times")
  # plt.xticks(bins, labels=[str(b) for b in bins])  # Ensure tick labels match bin edges
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  fig.savefig("preprocessing-time.pdf")
  plt.show()

def heatmap_operators(df):
  problems = df["problem"]

  ops = df[["num_op_max","num_op_eq", "num_op_reified_eq", "num_op_mul", "num_op_leq", "num_op_neq", "num_op_emod", "num_op_add", "num_op_reified_leq", "num_op_ediv", "num_op_gt", "num_op_min"]]

  tnf = [r"$x = \mathit{max}(y,z)$", r"$1 = (y=z)$", r"$x = (y=z)$", r"$x = y*z$", r"$1 = (y \leq z)$", r"$0 = (y=z)$", r"$x = y~\mathit{emod}~z$", r"$x = y + z$", r"$x = y \leq z$", r"$x = y / z$", r"$0 = (y \leq z)$", r"$x = \mathit{min}(y,z)$"]

  ops.columns = tnf

  ops = ops.div(ops.sum(axis=1), axis=0) * 100
  ops = ops.fillna(0)

  # Remove operators that never appear (columns where sum = 0)
  ops = ops.loc[:, ops.sum(axis=0) > 0]

  ops["problem"] = problems

  ops = ops.sort_values(by="problem")  # Group instances by problem
  ops = ops.drop(columns=["problem"])  # Remove problem column after sorting

  # Column sorting: order by total usage across all instances
  column_order = ops.sum(axis=0).sort_values(ascending=False).index
  ops = ops[column_order]  # Reorder columns

  fig, ax = plt.subplots(figsize=(8, 8))
  sns.heatmap(ops, cmap="coolwarm", xticklabels=True, yticklabels=False, ax=ax)
  # Identify problem group positions for labeling
  prev_problem = None
  start_idx = 0
  for i, problem in enumerate(problems[ops.index]):
    if problem != prev_problem and prev_problem is not None:
      mid_idx = (start_idx + i - 1) / 2  # Midpoint of the group
      ax.text(-0.1, mid_idx+0.2, prev_problem, ha="right", va="center", rotation=0)
      start_idx = i  # Update start index for new problem
    prev_problem = problem

  # Label the last problem group
  if prev_problem is not None:
    mid_idx = (start_idx + len(ops) - 1) / 2
    ax.text(-0.1, mid_idx, prev_problem, ha="right", va="center", rotation=0)

  yticks = []
  prev_problem = None
  for i, problem in enumerate(problems[ops.index]):  # Iterate in sorted order
    if problem != prev_problem:
      yticks.append(i)  # Place label in the middle of the group
      # ylabels.append(problem)
    prev_problem = problem

  plt.yticks(yticks)

  # plt.ylabel("Problem Groups")
  plt.xlabel("Operators")

  plt.title("Normalized Operator Usage Across Instances")
  fig.savefig("operators-heatmap.pdf", bbox_inches='tight')
  plt.show()
