import pandas as pd
import os

# Your local Excel files
excel_files = {
    "Friction": "experiment_results_friction.xlsx",
    "Hill": "experiment_results_hill.xlsx",
    "Mix": "experiment_results_mix.xlsx"
}

def calculate_all_percentages():
    print("=== Comprehensive Algorithm Performance Summary ===\n")
    
    # Track overall totals to give you an aggregate average across all maps
    overall = {
        'energy': {'Custom A*': [], 'Standard A*': [], 'Dijkstra': []},
        'dist_3d': {'Custom A*': [], 'Standard A*': [], 'Dijkstra': []},
        'total_time': {'Custom A*': [], 'Standard A*': [], 'Dijkstra': []}
    }
    
    metrics = {
        'energy': 'Energy (Joules)', 
        'dist_3d': 'Path Length (Meters)', 
        'total_time': 'Plan Time (Seconds)'
    }
    
    for map_type, filepath in excel_files.items():
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}")
            continue
            
        # Load the data
        df = pd.read_excel(filepath, engine='openpyxl')
        
        print(f"--- {map_type.upper()} MAP ---")
        
        for metric_col, metric_name in metrics.items():
            # Calculate mean for the specific metric
            avg_metric = df.groupby('algorithm')[metric_col].mean()
            
            try:
                custom_val = avg_metric['Custom A*']
                std_val = avg_metric['Standard A*']
                dijkstra_val = avg_metric['Dijkstra']
                
                # Store for overall average
                overall[metric_col]['Custom A*'].append(custom_val)
                overall[metric_col]['Standard A*'].append(std_val)
                overall[metric_col]['Dijkstra'].append(dijkstra_val)
                
            except KeyError as e:
                print(f"Missing algorithm data in {map_type} map: {e}")
                continue
                
            # Calculate percentage increases relative to Custom A*
            std_pct = ((std_val - custom_val) / custom_val) * 100
            dijkstra_pct = ((dijkstra_val - custom_val) / custom_val) * 100
            
            # Formatting the output to show if it was an increase (+) or decrease (-)
            std_sign = "↑" if std_pct > 0 else "↓"
            dijk_sign = "↑" if dijkstra_pct > 0 else "↓"
            
            print(f"  {metric_name}:")
            print(f"    Custom A*:   {custom_val:.2f} (Baseline)")
            print(f"    Standard A*: {std_val:.2f} (~{abs(std_pct):.0f}% {std_sign})")
            print(f"    Dijkstra:    {dijkstra_val:.2f} (~{abs(dijkstra_pct):.0f}% {dijk_sign})")
        print("\n")

    # ==========================================
    # OVERALL AGGREGATE CALCULATION
    # ==========================================
    print("======================================================")
    print("🏆 OVERALL AVERAGE ACROSS ALL MAPS (For your Table) 🏆")
    print("======================================================")
    
    for metric_col, metric_name in metrics.items():
        custom_list = overall[metric_col]['Custom A*']
        std_list = overall[metric_col]['Standard A*']
        dijkstra_list = overall[metric_col]['Dijkstra']
        
        if custom_list:
            avg_all_custom = sum(custom_list) / len(custom_list)
            avg_all_std = sum(std_list) / len(std_list)
            avg_all_dijkstra = sum(dijkstra_list) / len(dijkstra_list)
            
            agg_std_pct = ((avg_all_std - avg_all_custom) / avg_all_custom) * 100
            agg_dijkstra_pct = ((avg_all_dijkstra - avg_all_custom) / avg_all_custom) * 100
            
            std_sign = "↑" if agg_std_pct > 0 else "↓"
            dijk_sign = "↑" if agg_dijkstra_pct > 0 else "↓"
            
            print(f"{metric_name}:")
            print(f"  Custom A*:   Baseline")
            print(f"  Standard A*: ~{abs(agg_std_pct):.0f}% {std_sign}")
            print(f"  Dijkstra:    ~{abs(agg_dijkstra_pct):.0f}% {dijk_sign}\n")

if __name__ == "__main__":
    calculate_all_percentages()