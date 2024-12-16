import os
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict

def parse_timings_file(filename):
    """
    Parse a single timings file and extract timing data.
    
    Expected format in each file:
    A series of blocks introduced by:
        Timings{0}> ---------------------------------------------
        Timings{0}>      Timing results for X nodes:
        Timings{0}> ---------------------------------------------
        
    Followed by timing entries like:
        Timings{0}> updateParticle...... Wall max =    7.15055
        Timings{0}>                      Wall avg =    3.96428
        Timings{0}>                      Wall min =    2.59849
        
    Or sometimes a "Wall tot" line:
        Timings{0}> particleBC.......... Wall tot =  3.008e-06
    
    Each block seems to represent a single run's results for a certain number of nodes.
    The file may contain multiple runs (blocks) for the same number of nodes.
    
    Returns:
        A dict keyed by number_of_nodes, whose value is a dictionary mapping:
           "timer_name" -> {"max": float, "avg": float, "min": float, "tot": float (if present)}
        For timers without certain fields, those fields won't appear.
    """
    results = {}
    current_nodes = None
    current_timer = None
    timer_data = {}
    
    # Regex patterns to identify lines
    nodes_pattern = re.compile(r'Timing results for (\d+) nodes:')
    timer_name_pattern = re.compile(r'^Timings\{0\}>\s+([A-Za-z0-9_\.]+)\.*\s+Wall\s+(max|tot|avg|min)\s*=\s*(\S+)')
    timer_cont_pattern = re.compile(r'^Timings\{0\}>\s+Wall\s+(max|avg|min)\s*=\s*(\S+)')
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Check if line announces a new block of timing results
            match_nodes = nodes_pattern.search(line)
            if match_nodes:
                # Start a new results block for a given node count
                current_nodes = int(match_nodes.group(1))
                if current_nodes not in results:
                    results[current_nodes] = defaultdict(list)
                continue
            
            # Check for a new timer name line with one of the fields
            match_timer = timer_name_pattern.match(line)
            if match_timer:
                # This line defines a timer name and a field (max, avg, min, tot)
                timer_name = match_timer.group(1)
                field = match_timer.group(2)
                val = float(match_timer.group(3).replace('e', 'E'))  # ensure correct float parsing
                
                current_timer = timer_name
                timer_data = {field: val}
                
                # Store it temporarily
                # We will append after we read all lines for this timer
                continue
            
            # Check if line is a continuation line for the same timer
            match_cont = timer_cont_pattern.match(line)
            if match_cont and current_timer is not None:
                field = match_cont.group(1)
                val = float(match_cont.group(2).replace('e', 'E'))
                timer_data[field] = val
                continue
            
            # If we reach a blank line or a separating line and we have a current_timer:
            # that means we finished reading that timer block. Let's store it.
            if (line.startswith('Timings{0}>') == False and current_timer is not None) or \
               line.startswith('Timings{0}> ---------------------------------------------'):
                if current_timer and current_nodes is not None:
                    results[current_nodes][current_timer].append(timer_data)
                current_timer = None
                timer_data = {}
            
    # After finishing reading the file, if there's a last timer to store:
    if current_timer and current_nodes is not None:
        results[current_nodes][current_timer].append(timer_data)
    
    return results


def aggregate_timings(all_results):
    """
    Given a dictionary of:
       node_count -> {timer_name: [list_of_dicts_with_fields]}
    This function will average the values (max, avg, min, tot if present) across all runs for each timer.
    
    Returns:
        aggregated_results:
           node_count -> {timer_name: {"max": float, "avg": float, "min": float, "tot": float (if present)}}
    """
    aggregated = {}
    for nodes, timers in all_results.items():
        aggregated[nodes] = {}
        for timer_name, runs in timers.items():
            # runs is a list of dictionaries like: [{"max": ..., "avg": ...}, {...}, ...]
            # We want to average the fields across these runs
            fields = defaultdict(list)
            for run_data in runs:
                for k, v in run_data.items():
                    fields[k].append(v)
            
            averaged_fields = {k: np.mean(v) for k, v in fields.items()}
            aggregated[nodes][timer_name] = averaged_fields
    return aggregated


def load_all_timings_from_directory(directory):
    """
    Load and parse all files in the given directory (e.g., all *.txt).
    Aggregates them into a single dictionary.
    
    Returns aggregated results as described in aggregate_timings.
    """
    files = glob(os.path.join(directory, "*"))
    all_results = defaultdict(lambda: defaultdict(list))
    
    # Parse each file and combine results
    for filename in files:
        file_results = parse_timings_file(filename)
        # Merge into all_results
        for nodes, timers_dict in file_results.items():
            for timer_name, runs in timers_dict.items():
                all_results[nodes][timer_name].extend(runs)
                
    # Now aggregate
    aggregated = aggregate_timings(all_results)
    return aggregated


def plot_timing_vs_nodes(aggregated_results, timer_name, field='avg', show=True, savefig=None):
    """
    Given aggregated results of form:
        node_count -> {timer_name: {field: value}}
    Plot the chosen field (e.g., 'avg') of the given timer_name against the node_count.
    
    Params:
        aggregated_results: dict of aggregated timings from aggregate_timings
        timer_name: the name of the timer to plot (e.g. 'updateParticle', 'locateParticles')
        field: which field to plot ('avg', 'max', 'min', 'tot')
        show: whether to display the plot
        savefig: if given a filename, will save the figure instead of or in addition to showing
    """
    node_counts = []
    values = []
    
    for nodes in sorted(aggregated_results.keys()):
        timers = aggregated_results[nodes]
        if timer_name in timers and field in timers[timer_name]:
            node_counts.append(nodes)
            values.append(timers[timer_name][field])
    
    plt.figure(figsize=(8,6))
    plt.plot(node_counts, values, marker='o', linestyle='-', label=f'{timer_name} ({field})')
    plt.xlabel('Number of Nodes')
    plt.ylabel(f'{timer_name} ({field}) [Wall time]')
    plt.title(f'{timer_name} {field} vs. Number of Nodes')
    plt.grid(True)
    plt.legend()
    if savefig:
        plt.savefig(savefig)
    if show:
        plt.show()


def example_usage():
    # Example usage:
    # Suppose we have a directory "timings_logs" with multiple files.
    # aggregated_results = load_all_timings_from_directory("timings_logs")
    #
    # Now we can plot the average updateParticle time vs nodes:
    # plot_timing_vs_nodes(aggregated_results, "updateParticle", field='avg', savefig='updateParticle_vs_nodes.png')
    #
    # Similarly, we can do for locateParticles or other timers.
    pass


if __name__ == "__main__":
    # For actual use, modify the directory path as needed:
    # aggregated_results = load_all_timings_from_directory("path_to_your_directory")
    # plot_timing_vs_nodes(aggregated_results, "updateParticle", field='avg', show=True)
    pass