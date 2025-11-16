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
from statistics import mean, median, stdev

# A tentative to have unique experiment names.
def make_uid(config, arch, fixpoint, wac1_threshold, mzn_solver, version, machine, cores, timeout_ms, subproblems_power, subproblems_factor, or_nodes, threads_per_block, search, eps_val, eps_var, seed):
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
  if fixpoint != "":
    uid += "_" + fixpoint.lower()
    if fixpoint == "wac1":
      uid += "_" + str(int(wac1_threshold))
  if 'java11' in config:
    uid += '_java11'
  if mzn_solver == 'turbo.gpu.release':
    if int(subproblems_factor) != 30:
      uid += '_' + str(int(subproblems_factor)) + "sf"
    if int(subproblems_power) > 0:
      uid += '_' + str(int(subproblems_power)) + "sub"
    if int(or_nodes) != 0:
      uid += '_' + str(int(or_nodes)) + "blk"
    if cores > 1 and arch.lower() == 'hybrid':
      uid += "_" + str(int(cores))
    if int(threads_per_block) != 256:
      uid += '_' + str(int(threads_per_block)) + "TPB"
    if 'noatomics' in config:
      uid += '_noatomics'
    if 'globalmem' in config:
      uid += '_globalmem'
    if '_disable_simplify' in config:
      uid += '_disable_simplify'
    if '_disable_prop_removal' in config:
      uid += '_no_pr'
    if '_force_ternarize' in config:
      uid += '_force_ternarize'
    if '_ipc' in config:
      uid += '_ipc'
  else:
    if or_nodes > 1:
      uid += '_' + str(int(or_nodes)) + "threads"
    if cores > 1:
      uid += "_" + str(int(cores)) + "cores"
  if search == "free":
    uid += "_free"
  if isinstance(eps_var, str) and eps_var != "" and (eps_var != "default" or eps_val != "default"):
    uid += f"_{eps_var}"
    if eps_var == "random":
      uid += f"{seed}"
    uid += f"_{eps_val}"
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
    df.rename(columns={"subfactor": "subproblems_factor"}) # due to a mistake in the naming of that column.
    if "java11" in e:
      df['configuration'] = df['configuration'].apply(lambda x: x + "_java11")
    # Find how many subproblems are running in parallel.
    if 'or_nodes' not in df:
      if 'threads' in df:
        df['or_nodes'] = df['threads']
      else:
        df['or_nodes'] = df['configuration'].apply(lambda x: 1) # We suppose it is only 1 thread if the column is missing.
    if 'threads_per_block' not in df:
      if 'and_nodes' in df:
        df['threads_per_block'] = df['and_nodes']
      else:
        df['threads_per_block'] = 1
    if 'version' not in df:
      df['version'] = df['configuration'].apply(determine_version)
    else:
      df['version'] = df.apply(lambda row: determine_version(row["configuration"]) if not isinstance(row['version'], str) else row['version'], axis=1)
    if 'machine' not in df:
      df['machine'] = os.path.basename(os.path.dirname(e))
    if 'timeout_ms' not in df:
      df['timeout_ms'] = "300000"
    if 'num_blocks' not in df:
      df['num_blocks'] = df['or_nodes']
    if 'preprocessing_time' not in df:
      df['preprocessing_time'] = 0.0
    # else:
      # df['timeout_ms'] = df.apply(lambda row: "300000" if not isinstance(row['version'], int) else row['version'], axis=1)
    if 'failures' not in df:
      df['failures'] = 0
    if 'nSolutions' not in df:
      df['nSolutions'] = 0
    # estimating the number of nodes (lower bound).
    if 'nodes' not in df:
      df['nodes'] = (df['failures'] + df['nSolutions']) * 2 - 1
    if 'num_deductions' not in df:
      df['num_deductions'] = df['nodes']
    if 'cumulative_time_block_sec' not in df:
      df['cumulative_time_block_sec'] = df['num_blocks'] * (df['timeout_ms'].astype(float) / 1000.0)
    if 'deductions_per_block_second' not in df:
      df['deductions_per_block_second'] = df['num_deductions'] / df['num_blocks'] / df['cumulative_time_block_sec']
    if df[df['status'] == 'ERROR'].shape[0] > 0:
      print(e, ': Number of erroneous rows: ', df[df['status'] == 'ERROR'].shape[0])
      print(e, df[df['status'] == 'ERROR']['data_file'])
      df = df[df['status'] != 'ERROR']
    failed_xps = df[(df['mzn_solver'] == "turbo.gpu.release") & df['or_nodes'].isna()]
    if len(failed_xps) > 0:
      failed_xps_path = f"failed_{Path(e).name}"
      failed_xps[['configuration','problem', 'model', 'data_file']].to_csv(failed_xps_path, index=False)
      df = df[(df['mzn_solver'] != "turbo.gpu.release") | (~df['or_nodes'].isna())]
      print(f"{e}: {len(failed_xps)} failed experiments using turbo.gpu.release have been removed (the faulty experiments have been stored in {failed_xps_path}).")
    if 'search' not in df:
      df['search'] = 'user_defined'
    if 'subproblems_factor' not in df:
      df['subproblems_factor'] = 30
    if 'best_obj_time' not in df:
      obj_df = pd.read_csv(os.path.splitext(e)[0] + "-objectives.csv")
      # Convert the objective column to numeric (in case it contains non-numeric values)
      obj_df['objective'] = pd.to_numeric(obj_df['objective'], errors='coerce')

      # Find the best objective (minimum or maximum depending on the problem type) and its time
      best_objective_time = (obj_df.groupby(['configuration', 'problem', 'data_file'])
                            .tail(1)
                            .loc[:, ['configuration', 'problem', 'data_file', 'time']]
                            .rename(columns={'time': 'best_obj_time'}))
      df = pd.merge(df, best_objective_time, on=['configuration', 'problem', 'data_file'], how='left')
    # print(df[(df['mzn_solver'] == "turbo.gpu.release") & df['threads_per_block'].isna()])
    # df = df[(df['mzn_solver'] != "turbo.gpu.release") | (~df['threads_per_block'].isna())]
    all_xp = pd.concat([df, all_xp], ignore_index=True)
  all_xp['version'] = all_xp['version'].apply(version.parse)
  all_xp['nodes'] = all_xp['nodes'].fillna(0).astype(int)
  all_xp['num_deductions'] = all_xp['num_deductions'].fillna(0).astype(int)
  all_xp['status'] = all_xp['status'].fillna("UNKNOWN").astype(str)
  if 'memory_configuration' not in all_xp:
    all_xp['memory_configuration'] = 'RAM'
  all_xp['arch'] = all_xp['arch'].fillna("").astype(str) if 'arch' in all_xp else ""
  all_xp['propagator_mem'] = all_xp['propagator_mem'].fillna(0).astype(int) if 'propagator_mem' in all_xp else 0
  all_xp['store_mem'] = all_xp['store_mem'].fillna(0).astype(int) if 'store_mem' in all_xp else 0
  all_xp['fixpoint_iterations'] = pd.to_numeric(all_xp['fixpoint_iterations'], errors='coerce').fillna(0).astype(int) if 'fixpoint_iterations' in all_xp else 0
  all_xp['eps_num_subproblems'] = pd.to_numeric(all_xp['eps_num_subproblems'], errors='coerce').fillna(1).astype(int) if 'eps_num_subproblems' in all_xp else 1
  all_xp['subproblems_power'] = all_xp.apply(lambda row: int(math.log2(row['eps_num_subproblems'])) if math.isnan(row['subproblems_power']) else row['subproblems_power'], axis=1)  if 'subproblems_power' in all_xp else 0
  all_xp['subproblems_power'] = all_xp['subproblems_power'].astype(int)
  all_xp['num_blocks_done'] = pd.to_numeric(all_xp['num_blocks_done'], errors='coerce').fillna(0).astype(int) if 'num_blocks_done' in all_xp else 0
  all_xp['hardware'] = all_xp['machine'].apply(determine_hardware)
  all_xp['cores'] = all_xp['cores'].fillna(1).astype(int)
  # For the fixpoint, we default it to "" or "ac1" if it is a Turbo solver.
  all_xp['fixpoint'] = all_xp['fixpoint'].fillna("").astype(str) if 'fixpoint' in all_xp else ""
  all_xp['fixpoint'] = all_xp.apply(lambda row: "ac1" if row['fixpoint'] == "" and (row['mzn_solver'] == 'turbo.gpu.release' or row['mzn_solver'] == "turbo.cpu.release") else row['fixpoint'], axis=1)
  all_xp['wac1_threshold'] = all_xp['wac1_threshold'].fillna(0).astype(int) if "wac1_threshold" in all_xp else ""
  all_xp['cores'] = all_xp['cores'].fillna(1).astype(int)
  if 'seed' not in all_xp:
    all_xp['seed'] = 0
  else:
    all_xp['seed'] = all_xp['seed'].fillna(0).astype(int)
  if 'eps_value_order' not in all_xp:
    all_xp['eps_value_order'] = 'default'
  if 'eps_var_order' not in all_xp:
    all_xp['eps_var_order'] = 'default'
  all_xp['subproblems_factor'] = all_xp['subproblems_factor'].fillna(30).astype(int)
  all_xp['uid'] = all_xp.apply(lambda row: make_uid(row['configuration'], row['arch'], row['fixpoint'], row['wac1_threshold'], row['mzn_solver'], row['version'], row['machine'], row['cores'], row['timeout_ms'], row['subproblems_power'], row['subproblems_factor'], row['or_nodes'], row['threads_per_block'], row['search'], row['eps_value_order'], row['eps_var_order'], row['seed']), axis=1)
  all_xp['short_uid'] = all_xp['uid'].apply(make_short_uid)
  if 'solveTime' in all_xp:
    all_xp['nodes_per_second'] = all_xp['nodes'] / (all_xp['solveTime'] - all_xp['preprocessing_time'])
    all_xp['deductions_per_second'] = all_xp['num_deductions'] / (all_xp['solveTime'] - all_xp['preprocessing_time'])
    all_xp['deductions_per_node'] = all_xp['num_deductions'] / all_xp['nodes']
    all_xp['fp_iterations_per_node'] = all_xp['fixpoint_iterations'] / all_xp['nodes']
    all_xp['fp_iterations_per_second'] = all_xp['fixpoint_iterations'] / (all_xp['solveTime'] - all_xp['preprocessing_time'])
    all_xp['normalized_nodes_per_second'] = 0
    all_xp['normalized_deductions_per_second'] = 0
    all_xp['normalized_deductions_per_node'] = 0
    all_xp['normalized_fp_iterations_per_second'] = 0
    all_xp['normalized_fp_iterations_per_node'] = 0
    all_xp['dive_time'] = all_xp['dive_time'].fillna(0).astype(float) if 'dive_time' in all_xp else 0.0
    all_xp['preprocessing_time'] = all_xp['preprocessing_time'].fillna(0).astype(float) if 'preprocessing_time' in all_xp else 0.0
    all_xp['subproblem_solve_time'] = all_xp['solveTime'] - all_xp['preprocessing_time'] - all_xp['dive_time']
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

  # print(solver_instance)

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

def plot_overall_result(df):
  grouped = df.groupby(['short_uid', 'status']).size().unstack(fill_value=0)
  grouped['OPTIMAL/UNSAT'] = grouped.get('OPTIMAL_SOLUTION', 0) + grouped.get('UNSATISFIABLE', 0)

  # Ensure 'SATISFIED' and 'UNKNOWN' columns exist, even if they are all zeros
  if 'SATISFIED' not in grouped.columns:
      grouped['SATISFIED'] = 0
  if 'UNKNOWN' not in grouped.columns:
      grouped['UNKNOWN'] = 0

  grouped = grouped[['OPTIMAL/UNSAT', 'SATISFIED', 'UNKNOWN']]

  # Sort the DataFrame by 'OPTIMAL/UNSAT' and then 'SATISFIED'
  grouped.sort_values(by=['OPTIMAL/UNSAT', 'SATISFIED'], ascending=[True, True], inplace=True)

  # Plot
  colors = {'OPTIMAL/UNSAT': 'green', 'SATISFIED': 'lightblue', 'UNKNOWN': 'orange'}
  ax = grouped.plot(kind='barh', stacked=True, color=[colors[col] for col in grouped.columns])
  plt.title('Problem Status by Configuration')
  plt.ylabel('Configuration')
  plt.xlabel('Number of Problems (' + str(int(df.shape[0] / grouped.shape[0])) + ')')
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
  plt.tight_layout()
  plt.show()

def determine_version(solver_info):
  if 'choco' in solver_info:
    return "4.10.15"
  elif 'ortools' in solver_info:
    return "9.9"
  else:
    return "unknown"

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

def comparison_table_md(df, uid1, uid2):
  df2 = normalize(df, uid1, uid2, 'nodes_per_second')
  df2 = normalize(df2, uid1, uid2, 'fp_iterations_per_second')
  df2 = normalize(df2, uid1, uid2, 'fp_iterations_per_node')
  df2 = normalize(df2, uid1, uid2, 'deductions_per_node')
  df2 = normalize(df2, uid1, uid2, 'propagator_mem')
  df2 = normalize(df2, uid1, uid2, 'store_mem')
  m1 = metrics_table(df2[df2['uid'] == uid1]).squeeze()
  m2 = metrics_table(df2[df2['uid'] == uid2]).squeeze()
  total_pb = df2[df2['uid'] == uid1].shape[0]
  print(f"| Metrics | Normalized average [0,100] | Δ v{m1['version']} | #best (_/{total_pb}) | Average | Δ v{m1['version']} | Median | Δ v{m1['version']} |")
  print("|---------|----------------------------|----------|--------------|---------|----------|--------|----------|")
  print_table_line(m1, m2, "Nodes per second", "nodes_per_second", "")
  # print_table_line(m1, m2, "Deductions per second", "deductions_per_second", "")
  print_table_line(m1, m2, "Deductions per node", "deductions_per_node", "")
  print_table_line(m1, m2, "Fixpoint iterations per second", "fp_iterations_per_second", "")
  print_table_line(m1, m2, "Fixpoint iterations per node", "fp_iterations", "")
  print_table_line(m1, m2, "Propagators memory", "propagator_mem_mb", "MB")
  print_table_line(m1, m2, "Variables store memory", "store_mem_kb", "KB")
  print("")
  print(f"| Metrics | Count | Δ v{m1['version']} |")
  print("|---------|-------|----------|")
  print(f"| #Problems at optimality | {m2['problem_optimal']} | {m1['problem_optimal']} | ")
  print(f"| #Problems satisfiable | {m2['problem_sat']} | {m1['problem_sat']}  |")
  print(f"| #Problems unknown | {m2['problem_unknown']} | {m1['problem_unknown']}  |")
  print(f"| #Problem with store in shared memory | {m2['problem_with_store_shared']} | {m1['problem_with_store_shared']}  |")
  print(f"| #Problem with prop in shared memory | {m2['problem_with_props_shared']} | {m1['problem_with_props_shared']}  |")
  print(f"| #Problems with IDLE SMs at timeout | {m2['idle_eps_workers']} | {m1['idle_eps_workers']} |")

def metrics_table(df):
  grouped = df.groupby(['uid'])

  # Calculate metrics
  metrics = grouped.agg(
    version=('version', 'first'),
    machine=('machine', 'first'),
    short_uid=('short_uid', 'first'),
    avg_nodes_per_second=('nodes_per_second', lambda x: x[x != 0].mean()),
    median_nodes_per_second=('nodes_per_second', lambda x: x[x != 0].median()),
    avg_normalized_nodes_per_second=('normalized_nodes_per_second', lambda x: x[x != 0].mean()),
    best_nodes_per_second=('normalized_nodes_per_second', lambda x: x[x >= 100.0].count()),
    avg_deductions_per_node=('deductions_per_node', lambda x: x[x != 0].mean()),
    median_deductions_per_node=('deductions_per_node', lambda x: x[x != 0].median()),
    avg_normalized_deductions_per_node=('normalized_deductions_per_node', lambda x: x[x != 0].mean()),
    median_normalized_deductions_per_node=('normalized_deductions_per_node', lambda x: x[x != 0].median()),
    best_deductions_per_node=('normalized_deductions_per_node', lambda x: x[x < 100.0].count()),
    avg_deductions_per_second=('deductions_per_second', lambda x: x[x != 0].mean()),
    median_deductions_per_second=('deductions_per_second', lambda x: x[x != 0].median()),
    avg_normalized_deductions_per_second=('normalized_deductions_per_second', lambda x: x[x != 0].mean()),
    median_normalized_deductions_per_second=('normalized_deductions_per_second', lambda x: x[x != 0].median()),
    best_deductions_per_second=('normalized_deductions_per_second', lambda x: x[x < 100.0].count()),
    avg_fp_iterations_per_second=('fp_iterations_per_second', 'mean'),
    median_fp_iterations_per_second=('fp_iterations_per_second', 'median'),
    avg_normalized_fp_iterations_per_second=('normalized_fp_iterations_per_second', 'mean'),
    best_fp_iterations_per_second=('normalized_fp_iterations_per_second', lambda x: x[x >= 100.0].count()),
    avg_fp_iterations=('fp_iterations_per_node', 'mean'),
    median_fp_iterations=('fp_iterations_per_node', 'median'),
    avg_normalized_fp_iterations=('normalized_fp_iterations_per_node', 'mean'),
    best_fp_iterations=('normalized_fp_iterations_per_node', lambda x: x[x < 100.0].count()),
    avg_propagator_mem_mb=('propagator_mem', lambda x: x.mean() / 1000000),
    median_propagator_mem_mb=('propagator_mem', lambda x: x.median() / 1000000),
    avg_normalized_propagator_mem_mb=('normalized_propagator_mem', 'mean'),
    best_propagator_mem_mb=('normalized_propagator_mem', lambda x: x[x < 100.0].count()),
    avg_store_mem_kb=('store_mem', lambda x: x.mean() / 1000),
    avg_normalized_store_mem_kb=('normalized_store_mem', 'mean'),
    best_store_mem_kb=('normalized_store_mem', lambda x: x[x < 100.0].count()),
    median_store_mem_kb=('store_mem', lambda x: x.median() / 1000),
    problem_optimal=('status', lambda x: (x == 'OPTIMAL_SOLUTION').sum() + (x == 'UNSATISFIABLE').sum()),
    problem_sat=('status', lambda x: (x == 'SATISFIED').sum()),
    problem_unknown=('status', lambda x: (x == 'UNKNOWN').sum()),
    problem_with_store_shared=('memory_configuration', lambda x: (x == "store_shared").sum()),
    problem_with_props_shared=('memory_configuration', lambda x: ((x == "store_pc_shared") | (x == "tcn_shared")).sum())
  )

  # Count problems with non-zero num_blocks_done and not solved to optimality or proven unsatisfiable
  condition = (df['num_blocks_done'] != 0) & (~df['status'].isin(['OPTIMAL_SOLUTION', 'UNSATISFIABLE']))
  idle_eps_workers = df[condition].groupby(['uid']).size().reset_index(name='idle_eps_workers')

  # Merge metrics with idle_eps_workers
  overall_metrics = metrics.merge(idle_eps_workers, on=['uid'], how='left').fillna(0)
  overall_metrics = overall_metrics.sort_values(by=['avg_nodes_per_second', 'version', 'machine'], ascending=[False, False, True])

  return overall_metrics

def compare_solvers(df, uid1, uid2, uid1_label = None, uid2_label = None):
    if uid1_label == None:
      uid1_label = uid1
    if uid2_label == None:
      uid2_label = uid2

    solvers_df = df[(df['uid'] == uid1) | (df['uid'] == uid2)]

    # Pivoting for 'objective', 'method', and 'status' columns
    pivot_df = solvers_df.pivot_table(index='model_data_file', columns='uid', values=['model', 'problem', 'data_file', 'objective', 'method', 'status'], aggfunc='first')

    # Compare objective values based on method and optimality status
    conditions = [
        # Error
        (pivot_df['method', uid1] != pivot_df['method', uid2]),

        # Solver 1 better
        ((pivot_df['status', uid1] != "UNKNOWN") & (pivot_df['status', uid2] == "UNKNOWN")) |
        ((pivot_df['method', uid1] == "minimize") & (pivot_df['objective', uid1] < pivot_df['objective', uid2])) |
        ((pivot_df['method', uid1] == "maximize") & (pivot_df['objective', uid1] > pivot_df['objective', uid2])) |
        ((pivot_df['method', uid1] == "satisfy") & (pivot_df['status', uid1] == "UNSATISFIABLE") & (pivot_df['status', uid2] != "UNSATISFIABLE")) |
        ((pivot_df['method', uid1] != "satisfy") & (pivot_df['objective', uid1] == pivot_df['objective', uid2]) & (pivot_df['status', uid1] == "OPTIMAL_SOLUTION") & (pivot_df['status', uid2] != "OPTIMAL_SOLUTION")),

        # Solver 2 better
        ((pivot_df['status', uid1] == "UNKNOWN") & (pivot_df['status', uid2] != "UNKNOWN")) |
        ((pivot_df['method', uid1] == "minimize") & (pivot_df['objective', uid1] > pivot_df['objective', uid2])) |
        ((pivot_df['method', uid1] == "maximize") & (pivot_df['objective', uid1] < pivot_df['objective', uid2])) |
        ((pivot_df['method', uid1] == "satisfy") & (pivot_df['status', uid1] != "UNSATISFIABLE") & (pivot_df['status', uid2] == "UNSATISFIABLE")) |
        ((pivot_df['method', uid1] != "satisfy") & (pivot_df['objective', uid1] == pivot_df['objective', uid2]) & (pivot_df['status', uid1] != "OPTIMAL_SOLUTION") & (pivot_df['status', uid2] == "OPTIMAL_SOLUTION")),

        # Equal
        (pivot_df['status', uid1] == pivot_df['status', uid2])
    ]

    choices = ['Error', f'{uid1_label} better', f'{uid2_label} better', 'Equal']

    pivot_df['Comparison'] = np.select(conditions, choices, default='Unknown')

    # Get problems with "Unknown" comparison (should not happen, this is for debugging).
    unknown_problems = pivot_df[pivot_df['Comparison'] == 'Unknown'].index.tolist()
    if unknown_problems:
        print(f"The comparison is 'Unknown' for the following problems: {', '.join(unknown_problems)}")

    error_problems = pivot_df[pivot_df['Comparison'] == 'Error'].index.tolist()
    if error_problems:
        print(f"The comparison is 'Error' for the following problems: {', '.join(error_problems)}")
    return pivot_df

def compare_solvers_pie_chart(df, uid1, uid2, uid1_label = None, uid2_label = None):
    """
    Compares the performance of two solvers based on objective value and optimality.

    Parameters:
    - df: DataFrame containing the data.
    - uid1: Name of the first solver (str).
    - uid2: Name of the second solver (str).

    Returns:
    - Displays a pie chart comparing the performance of the two solvers.
    """
    if uid1_label == None:
      uid1_label = uid1
    if uid2_label == None:
      uid2_label = uid2

    pivot_df = compare_solvers(df, uid1, uid2, uid1_label, uid2_label)
    # Get counts for each category
    category_counts = pivot_df['Comparison'].value_counts()
    print(category_counts)
    color_mapping = {
        f'{uid1_label} better': 'green',
        f'{uid2_label} better': 'orange',
        'Equal': (0.678, 0.847, 0.902), # light blue
        'Unknown': 'red',
        'Error': 'red'
    }
    colors = [color_mapping[cat] for cat in category_counts.index]
    plt.rcParams.update({'font.size': 20})
    # Plot pie chart
    fig = plt.figure(figsize=(10, 6))
    category_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=colors)
    # plt.title(f'Objective Value and Optimality Comparison between {uid1} and {uid2}')
    plt.ylabel('')
    fig.savefig(f"cmp-{uid1_label}-{uid2_label}.png")
    plt.show()
    return pivot_df

# List the problems on which the key is greater on the second solver
def list_problem_where_leq(df, key, uid1, uid2):
  df1 = df[df['uid'] == uid1]
  df2 = df[df['uid'] == uid2]
  # Merge the two dataframes on the 'model_data_file' to compare them
  comparison_df = pd.merge(df1[['model_data_file', key]], df2[['model_data_file', key]], on='model_data_file', suffixes=('_1', '_2'))
  # Find where key is greater on df2
  return comparison_df[comparison_df[key+"_1"] > comparison_df[key+"_2"]]

def block_idling_analysis(df, uids):
  print("uid | problems with blocks idling | avg proportion idling | median proportion idling | std proportion idling")
  print("--- | --- | --- | --- | ---")
  for uid in uids:
    df2 = df[df['uid'] == uid]
    df2 = df2[df2['num_blocks_done'] > 0]
    df2 = df2[(df2['status'] != 'OPTIMAL_SOLUTION') & (df2['status'] != 'UNSATISFIABLE')]

    df2['proportion_idling'] = df2['num_blocks_done'] / df2['num_blocks']

    print(f"{uid} | {df2.shape[0]} | {df2['proportion_idling'].mean():.2f} | {df2['proportion_idling'].median():.2f} | {np.std(df2['proportion_idling'], ddof=0):.2f}")

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


  df2 = df[df['num_blocks_done'] == 0]
  df2['proportion_fixpoint'] = df2['fixpoint_time'] / (df2['solveTime'] - df2['preprocessing_time'])
  print("Average proportion of time spent in fixpoint: ", df2['proportion_fixpoint'].mean())
  print("Median proportion of time spent in fixpoint: ", df2['proportion_fixpoint'].median())
  print("Standard deviation of proportion of time spent in fixpoint: ", np.std(df2['proportion_fixpoint'], ddof=0))

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
      if(row['model_data_file'] == "../data/mzn-challenge/2024/fox-geese-corn/foxgeesecorn.mzn - ../data/mzn-challenge/2024/fox-geese-corn/foxgeesecorn_17.dzn"):
        print(row['model_data_file'], row['num_blocks_done'])
      if num_blocks != 0:
          ax.text(
              total_time + 0.5,  # slightly to the right of the end of the bar
              idx,               # vertical position
              str(int(num_blocks)),  # convert to string for display
              va='center', ha='left', fontsize=8, color='black'
          )

  # Show plot
  plt.show()

def dive_solve_distribution(df, plot=False):
  time_columns = [
    "preprocessing_time",
    "dive_time",
    "subproblem_solve_time"
  ]

  df2 = df[df['num_blocks_done'] == 0]
  df2['proportion_diving'] = df2['dive_time'] / (df2['dive_time'] + df2['subproblem_solve_time'])
  print("Average proportion of time spent in diving: ", df2['proportion_diving'].mean())
  print("Median proportion of time spent in diving: ", df2['proportion_diving'].median())
  print("Standard deviation of proportion of time spent in diving: ", np.std(df2['proportion_diving'], ddof=0))

  if plot:
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
    plt.title("Time Distribution in Solver Components for Each Problem")
    plt.legend(title="Time Component", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Show plot
    plt.show()

def boxplot_preprocessing_components(df, components):
  pass

def boxplot_tcn_increase(df, ref_vars, ref_cons):
  # parsed_variables
  # tnf_variables
  # variables_after_simplification

  pass

def analyse_tnf_per_problem(df, logy, source_vars, source_cons, target_vars, target_cons, list_problems=False):
  if list_problems:
    print(f"| Problem | Data | #Vars | #Vars (TNF) | #Constraints | #Constraints (TNF) | Preprocessing (sec) |")
    print("|----------|------|-------|-------------|--------------|--------------------|---------------------|")
  all_vars_increase = []
  all_cons_increase = []
  preprocessing_times = []
  leq_10x_increase = 0
  for index, row in df.iterrows():
    vars_increase = row[target_vars] / row[source_vars]
    cons_increase = row[target_cons] / row[source_cons]
    if vars_increase <= 10. and cons_increase <= 10.:
      leq_10x_increase += 1
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

# def preprocessing_contribution(df):

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

def heatmap_solver_comparison(df, source_solver, target_solvers, global_cons = None):
    comparisons = []
    for target in target_solvers:
        comparisons.append(compare_solvers(df, source_solver[0], target[0], source_solver[1], target[1]))

    # Build the problem list from the comparison pivot
    problems = comparisons[0][('problem', source_solver[0])]

    heatmap_data = pd.DataFrame(index=problems.index)

    color_map = {
        f"{source_solver[1]} better": 1,
        "Equal": 0,
    }
    for target in target_solvers:
        color_map[f"{target[1]} better"] = -1

    for comparison, target in zip(comparisons, target_solvers):
        mapped = comparison['Comparison'].map(color_map)
        heatmap_data[target[1]] = mapped

    # Now sort by problems
    heatmap_data["problem"] = problems.values
    heatmap_data = heatmap_data.sort_values("problem")
    heatmap_data = heatmap_data.drop(columns=["problem"])

    # --- Custom colormap ---
    colors = [
        (1.0, 0.5, 0.5),  # light red for -1
        (1.0, 1.0, 1.0),  # white for 0
        (0.5, 0.8, 0.5),  # light green for +1
    ]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

    # --- Plotting ---
    plotwidth = 2 * len(target_solvers)
    if global_cons is not None:
      plotwidth += 4
    fig, ax = plt.subplots(figsize=(plotwidth, 9))
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        vmin=-1, vmax=1,
        cbar=False,
        xticklabels=True,
        yticklabels=False,
        ax=ax,
        linewidths=0.5,
        linecolor='lightgrey'
    )
    ax.set_ylabel("")  # <- no label
    # Add problem group labels
    prev_problem = None
    start_idx = 0
    for i, idx in enumerate(heatmap_data.index):
        current_problem = problems.loc[idx]
        if current_problem != prev_problem and prev_problem is not None:
            mid_idx = (start_idx + i - 1) / 2
            ax.text(-0.1, mid_idx + 0.2, prev_problem, ha="right", va="center", rotation=0, fontsize=9)
            # Add global constraints for the previous group
            constraint = global_cons.loc[global_cons['problem'] == prev_problem, 'globals'].values
            if constraint.size > 0 and isinstance(constraint[0], str) and constraint[0] != "":
                wrapped = "\n".join(textwrap.wrap(constraint[0], width=40))
                ax.text(len(target_solvers) + 0.1, mid_idx + 0.5, wrapped,
                        ha="left", va="center", rotation=0, fontsize=8, color='dimgray')
            start_idx = i
        prev_problem = current_problem

    if prev_problem is not None:
        mid_idx = (start_idx + len(heatmap_data) - 1) / 2
        ax.text(-0.1, mid_idx + 0.2, prev_problem, ha="right", va="center", rotation=0, fontsize=9)
        constraint = global_cons.loc[global_cons['problem'] == prev_problem, 'globals'].values
        if constraint.size > 0 and isinstance(constraint[0], str) and constraint[0] != "":
            wrapped = "\n".join(textwrap.wrap(constraint[0], width=40))
            ax.text(len(target_solvers) + 0.1, mid_idx + 0.5, wrapped,
                    ha="left", va="center", rotation=0, fontsize=8, color='dimgray')

    yticks = []
    prev_problem = None
    for i, idx in enumerate(heatmap_data.index):  # Iterate in sorted order
      current_problem = problems.loc[idx]
      if current_problem != prev_problem:
        yticks.append(i)  # Place label in the middle of the group
      prev_problem = current_problem
    plt.yticks(yticks)

    plt.xlabel("Target Solvers", fontsize=12)
    plt.title(f"Comparison of {source_solver[1]} against Target Solvers", fontsize=14)
    plt.tight_layout()
    plt.show()

# Helper functions to determine scoring criteria
def solved(row):
    """Determine if a solver has successfully solved the problem."""
    return row['status'] in ['SATISFIED', 'OPTIMAL_SOLUTION']

def optimal(row):
    """Determine if the solver found the optimal solution."""
    return row['status'] == 'OPTIMAL_SOLUTION'

def quality(row):
    """Return the quality of the solution (objective value)."""
    try:
        return float(row['objective'])
    except ValueError:
        return np.nan

def compute_score(time_s, time_s_prime):
    """Calculate the score for indistinguishable answers based on time."""
    if time_s == 0 and time_s_prime == 0:
        return 0.5
    return time_s_prime / (time_s_prime + time_s)

def mzn_is_better(row_A, row_B):
  if row_A['status'] != 'UNKNOWN' and row_B['status'] == 'UNKNOWN':
    return True
  if row_A['status'] == 'OPTIMAL_SOLUTION' and row_B['status'] != 'OPTIMAL_SOLUTION':
    return True
  if row_A['status'] == 'UNSATISFIABLE' and row_B['status'] != 'UNSATISFIABLE':
    return True
  if row_A['status'] == 'SATISFIED' and row_B['status'] == 'SATISFIED':
    if row_A['method'] == 'minimize' and row_A['objective'] < row_B['objective']:
      return True
    elif row_A['method'] == 'maximize' and row_A['objective'] > row_B['objective']:
      return True
  return False

def uid_problem_row(df, uid, P):
  row = df[(df['uid'] == uid) & (df['model_data_file'] == P)]
  if row.empty:
    print(f"Warning: Missing data for solvers {uid} on problem {P}.")
  if len(row) != 1:
    print(f"Warning: Expected one row for solvers {uid} and on problem {P}, but got {len(row)} (using the first one).")
  return row.iloc[0] if not row.empty else None

def minizinc_challenge_score(df, solvers_uid=[]):
  # Initialize a dictionary to store Borda scores for each solver
  scores = {}
  solvers = df['uid'].unique() if len(solvers_uid) == 0 else solvers_uid
  # Iterate over each pair of solvers (A, B)
  for A in solvers:
    for B in solvers:
      if A == B:
        continue  # Skip comparing the solver to itself
      for P in df['model_data_file'].unique():
        scores.setdefault((A, P), 0.0)
        row_A = uid_problem_row(df, A, P)
        row_B = uid_problem_row(df, B, P)
        if row_A is None or row_B is None:
          continue

        # No point awarded if the status is unknown.
        if row_A['status'] == 'UNKNOWN':
          continue

        if mzn_is_better(row_A, row_B):
          scores[(A,P)] += 1
        elif row_A['status'] == 'SATISFIED' and row_B['status'] == 'SATISFIED' and row_A['method'] in ['minimize', 'maximize'] and row_A['objective'] == row_B['objective']:
          if pd.isnull(row_A['best_obj_time']):
            print(f"Warning: Missing time data for solvers {A} on problem {P}.")
            continue
          if pd.isnull(row_B['best_obj_time']):
            print(f"Warning: Missing time data for solvers {B} on problem {P}.")
            continue
          time_A = float(row_A['best_obj_time'])
          time_B = float(row_B['best_obj_time'])
          scores[(A,P)] = time_B / (time_A + time_B) if (time_A + time_B) != 0 else 0.5

  return scores

def scores_summary(scoring_method, scores):
  # Initialize a dictionary to store the total score for each solver
  total_scores = {}
  for (solver, _), score in scores.items():
    if solver not in total_scores:
      total_scores[solver] = 0.0
    total_scores[solver] += score

  # Sort the solvers by their total score
  sorted_solvers = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)

  return pd.DataFrame(sorted_solvers, columns=['solver', scoring_method + ' score'])

def best_solutions_per_instance(df, solvers_uid):
  best_solutions = {}
  for P in df['model_data_file'].unique():
    best_solutions[P] = (None, 'UNKNOWN')
    for A in solvers_uid:
      row_A = uid_problem_row(df, A, P)
      if row_A is None or row_A['status'] == 'UNKNOWN':
        continue
      if row_A['status'] == 'OPTIMAL_SOLUTION' or best_solutions[P][0] is None:
        if row_A['status'] == 'SATISFIED':
          best_solutions[P] = (row_A['objective'], row_A['status'])
      elif row_A['method'] == "minimize" and best_solutions[P][0] > row_A['objective']:
        best_solutions[P] = (row_A['objective'], row_A['status'])
      elif row_A['method'] == "maximize" and best_solutions[P][0] < row_A['objective']:
        best_solutions[P] = (row_A['objective'], row_A['status'])
  return best_solutions

def xcsp3_challenge_score(df, solvers_uid=[]):
  # Initialize a dictionary to store XCSP3 scores for each solver
  scores = {}
  solvers = df['uid'].unique() if len(solvers_uid) == 0 else solvers_uid

  # Compute the best solutions for each instance.
  best_solutions = best_solutions_per_instance(df, solvers)

  # Compute the scores for each solver on each instance.
  for A in solvers:
    for P in df['model_data_file'].unique():
      scores.setdefault((A, P), 0.0)
      row_A = uid_problem_row(df, A, P)
      if row_A is None or row_A['status'] == 'UNKNOWN':
        continue

      if row_A['method'] == 'satisfy':
        if row_A['status'] == 'SATISFIED':
          scores[(A,P)] += 1
        elif row_A['status'] == 'UNSATISFIABLE':
          scores[(A,P)] += 1
      else:
        if row_A['status'] == 'UNSATISFIABLE':
          scores[(A,P)] += 1
        elif row_A['status'] == 'OPTIMAL_SOLUTION':
          scores[(A,P)] += 1
        elif row_A['status'] == 'SATISFIED':
          if row_A['objective'] == best_solutions[P][0]:
            scores[(A,P)] += 0.5 if best_solutions[P][1] == 'OPTIMAL_SOLUTION' else 1
  return scores

def analyze_fp_iterations_on_backtrack(df):
  df = df[df['depth'] != 0]
  df['original_index'] = df.index
  df_sorted = df.sort_values(by=['blockIdx', 'original_index'])

  # Lists to store fp_iterations
  fp_after_backtrack_per_blocks = {}
  fp_no_backtrack_per_blocks = {}
  fp_after_backtrack = []
  fp_no_backtrack = []

  # Track previous depth per blockIdx
  previous_depths = {}

  for _, row in df_sorted.iterrows():
    block = row['blockIdx']
    depth = row['depth']
    fp = float(row['fp_iterations'])
    if block not in fp_after_backtrack_per_blocks:
      fp_after_backtrack_per_blocks[block] = []
      fp_no_backtrack_per_blocks[block] = []

    if block not in previous_depths or depth > previous_depths[block]:
      fp_no_backtrack.append(fp)
      fp_no_backtrack_per_blocks[block].append(fp)
    else:
      fp_after_backtrack.append(fp)
      fp_after_backtrack_per_blocks[block].append(fp)
    previous_depths[block] = depth

  print(f'Number of backtracks/total: {len(fp_after_backtrack)}/{len(fp_no_backtrack)}\n')

  if len(fp_no_backtrack_per_blocks) > 1:
    print("Standard deviation of average fp_iterations per block:")
    print("  After backtrack:", stdev([mean(fps) for fps in fp_after_backtrack_per_blocks.values() if fps]))
    print("  Without backtrack:", stdev([mean(fps) for fps in fp_no_backtrack_per_blocks.values() if fps]))

    print("\nStandard deviation of median fp_iterations per block:")
    print("  After backtrack:", stdev([median(fps) for fps in fp_after_backtrack_per_blocks.values() if fps]))
    print("  Without backtrack:", stdev([median(fps) for fps in fp_no_backtrack_per_blocks.values() if fps]))

  print(f"\n\n  Average fp_iterations: {mean(fp_after_backtrack + fp_no_backtrack)}")
  print(f"  Median fp_iterations: {median(fp_after_backtrack + fp_no_backtrack)}")

  print("\nAfter a backtrack:")
  print(f"  Average fp_iterations: {mean(fp_after_backtrack)}")
  print(f"  Median fp_iterations: {median(fp_after_backtrack)}")
  print(f"  Standard deviation fp_iterations: {stdev(fp_after_backtrack)}")
  print(f"  Min fp_iterations: {min(fp_after_backtrack)}")
  print(f"  Max fp_iterations: {max(fp_after_backtrack)}")

  print("\nWithout a backtrack:")
  print(f"  Average fp_iterations: {mean(fp_no_backtrack)}")
  print(f"  Median fp_iterations: {median(fp_no_backtrack)}")
  print(f"  Standard deviation fp_iterations: {stdev(fp_no_backtrack)}")
  print(f"  Min fp_iterations: {min(fp_no_backtrack)}")
  print(f"  Max fp_iterations: {max(fp_no_backtrack)}")


def plot_mem_distribution(df):
  mem_columns = [
    "store_mem_kb",
    "propagator_mem_kb"
  ]

  df["store_mem_kb"] = df["store_mem"] / 1000
  df["propagator_mem_kb"] = df["propagator_mem"] / 1000
  df["estimated_copy_strat_mb"] = (df["store_mem_kb"] + df["propagator_mem_kb"]) * df["peakDepth"] * df["num_blocks"]
  df["estimated_copy_strat_mb"] = df["estimated_copy_strat_mb"].astype(float) / 1000.0 / 1000.0

  df.sort_values(by="problem_uid", ascending=False, inplace=True)
  # Set the problem names as index
  df.set_index("problem_uid", inplace=True)

  num_row = df.shape[0]

  # Plot
  colors = ["#1f77b4", "#ff7f0e"]
  ax = df[mem_columns].plot(kind='barh', stacked=True, figsize=(10, num_row / 2.5), color=colors)

  # Add labels and title
  plt.xlabel("Memory (KB)")
  plt.xscale('log')
  plt.ylabel("Problem")
  plt.title("Memory Distribution for Each Problem (and estimated memory (MB) for full copying)")
  plt.legend(title="Memory Component", bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.tight_layout()

  # Add label indicating the estimated memory taken by a full copying restoration strategy.
  for idx, (index, row) in enumerate(df.iterrows()):
    total_mem = row[mem_columns].sum()
    if total_mem != 0:
      ax.text(
        total_mem + 0.5,  # slightly to the right of the end of the bar
        idx,               # vertical position
        ('%.1f' % row["estimated_copy_strat_mb"]) + ' | ' + str(int(row["peakDepth"])),  # convert to string for display
        va='center', ha='left', fontsize=10, color='black'
      )

  plt.show()
