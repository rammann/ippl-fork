import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_timing_file(filepath):
    # Initialize data storage
    data = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # Extract number of nodes from directory name
    n_nodes = int(filepath.parent.name.split('_n')[1])
    
    # Parse the timing data
    reading_totals = False
    reading_averages = False
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('='): 
            continue
            
        if 'Wall tot' in line:
            reading_totals = True
            reading_averages = False
            continue
            
        if 'Wall max' in line:
            reading_totals = False
            reading_averages = True
            continue
            
        parts = line.split()
        operation = parts[0].strip('.')
        
        if reading_totals and len(parts) >= 3:
            data[operation] = {
                'nodes': n_nodes,
                'wall_tot': float(parts[-1])
            }
        elif reading_averages and len(parts) >= 5:
            if operation not in data:
                data[operation] = {'nodes': n_nodes}
            data[operation].update({
                'wall_max': float(parts[-3]),
                'wall_min': float(parts[-2]),
                'wall_avg': float(parts[-1])
            })
    
    return data

def collect_all_timing_data(base_path):
    all_data = []
    
    # Walk through all N1_* and N2_* directories
    for directory in sorted(Path(base_path).glob('N[12]_n*')):
        timing_file = directory / 'timings0.dat'
        if timing_file.exists():
            data = parse_timing_file(timing_file)
            all_data.append(data)
    
    return all_data

def create_timing_dataframe(all_data):
    # Flatten the data structure into a format suitable for pandas
    rows = []
    for data_dict in all_data:
        for operation, metrics in data_dict.items():
            row = {'operation': operation, **metrics}
            rows.append(row)
    
    return pd.DataFrame(rows)

def plot_scaling_analysis(df):
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Plot wall average time vs number of nodes
    sns.scatterplot(data=df, x='nodes', y='wall_avg', hue='operation', style='operation')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Average Wall Time Scaling')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Wall Time (s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('scaling_analysis.png')
    plt.close()

def plot_operation_breakdown(df, operations=None):
    """
    Create a stacked bar chart for specified operations.
    
    Args:
        df: DataFrame containing the timing data
        operations: List of operation names to plot. If None, plots all operations.
    """
    plt.figure(figsize=(15, 8))
    
    # Filter the DataFrame for specified operations
    if operations is not None:
        df_filtered = df[df['operation'].isin(operations)]
    else:
        df_filtered = df
    
    # Pivot the filtered data
    pivot_data = df_filtered.pivot(index='nodes', columns='operation', values='wall_avg')
    pivot_data.plot(kind='bar', stacked=True)
    
    plt.title('Operation Time Breakdown by Node Count')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Wall Time (s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if operations is not None:
        plt.savefig('operation_breakdown_{}.png'.format('_'.join(operations)))
    else:
        plt.savefig('operation_breakdown.png')
    plt.close()

def plot_operation_grouped(df, operations=None, exclude_operations=None):
    """
    Create a grouped bar chart for specified operations, excluding certain operations if needed.
    
    Args:
        df: DataFrame containing the timing data.
        operations: List of operation names to plot. If None, plots all operations.
        exclude_operations: List of operation names to exclude from the plot.
    """
    plt.figure(figsize=(15, 8))
    
    # Filter the DataFrame for specified operations
    if operations is not None:
        df_filtered = df[df['operation'].isin(operations)]
    else:
        df_filtered = df
    
    # Exclude specified operations
    if exclude_operations is not None:
        df_filtered = df_filtered[~df_filtered['operation'].isin(exclude_operations)]
    
    # Pivot the filtered data
    pivot_data = df_filtered.pivot(index='nodes', columns='operation', values='wall_avg')
    
    # Plot using a grouped bar chart
    ax = pivot_data.plot(kind='bar', width=0.8)
    
    plt.title('Operation Time Breakdown by Node Count')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Wall Time (s)')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('operation_grouped.png')
    plt.close()

def main():
    # Assuming you're in the parent directory containing all N*_n* directories
    base_path = '.'
    
    # Collect all timing data
    all_data = collect_all_timing_data(base_path)
    
    # Create a pandas DataFrame
    df = create_timing_dataframe(all_data)
    
    # Generate plots
    print("Available operations:", sorted(df['operation'].unique()))
    plot_scaling_analysis(df)
    plot_operation_grouped(df, operations=['block_partition', 'build_tree', 'complete_region', 'complete_tree', 'getNumParticlesInOc', 'linearise_octants', 'orthotree_build', 'partition'])
    plot_operation_breakdown(df, operations=['orthotree_build'])
    plot_operation_breakdown(df, operations=['build_tree'])
    plot_operation_breakdown(df, operations=['getNumParticlesInOc'])
    
    # Save the processed data
    df.to_csv('timing_analysis.csv', index=False)
    
    # Print summary statistics
    print("\nSummary of timing data:")
    print(df.groupby('operation')['wall_avg'].describe())

if __name__ == '__main__':
    main()
