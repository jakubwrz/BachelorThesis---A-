import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set visualization style
sns.set_theme(style="whitegrid")

excel_files = {
    "Friction": "experiment_results_friction.xlsx",
    "Hill": "experiment_results_hill.xlsx",
    "Mix": "experiment_results_mix.xlsx"
}

def generate_stats_and_graphs():
    for map_type, filepath in excel_files.items():
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}. Please ensure it is in the same directory as this script.")
            continue

        # Load the data directly from the Excel file
        df = pd.read_excel(filepath, engine='openpyxl')

        # ==========================================
        # 1. PRINT STATISTICS TO CONSOLE
        # ==========================================
        print(f"\n" + "="*60)
        print(f"📊 STATISTICS FOR: {map_type.upper()} MAP")
        print("="*60)
        
        metrics = ['energy', 'total_time', 'dist_2d', 'dist_3d']
        stats_mean = df.groupby('algorithm')[metrics].mean()
        stats_std = df.groupby('algorithm')[metrics].std()
        
        print("\n🟢 MEAN VALUES (Averages):")
        print(stats_mean.round(4).to_string())
        
        print("\n🟡 STANDARD DEVIATION (Variance across runs):")
        print(stats_std.round(4).to_string())
        
        # Define matching colors and markers for the graphs
        palette = {"Custom A*": "#2ecc71", "Standard A*": "#3498db", "Dijkstra": "#e74c3c"}
        markers = {"Custom A*": "o", "Standard A*": "s", "Dijkstra": "^"}

        # ==========================================
        # 2. GENERATE BOXPLOTS (Original Graphs)
        # ==========================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Algorithm Performance Comparison - {map_type} Map', fontsize=16, fontweight='bold')

        sns.boxplot(ax=axes[0, 0], data=df, x='algorithm', y='energy', palette=palette, hue='algorithm', legend=False)
        axes[0, 0].set_title('Energy Consumption (Joules)', fontweight='bold')
        axes[0, 0].set_ylabel('Total Energy')
        axes[0, 0].set_xlabel('')

        sns.boxplot(ax=axes[0, 1], data=df, x='algorithm', y='total_time', palette=palette, hue='algorithm', legend=False)
        axes[0, 1].set_title('Total Calculation Time (Seconds)', fontweight='bold')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].set_xlabel('')

        sns.boxplot(ax=axes[1, 0], data=df, x='algorithm', y='dist_2d', palette=palette, hue='algorithm', legend=False)
        axes[1, 0].set_title('2D Distance (Meters)', fontweight='bold')
        axes[1, 0].set_ylabel('Distance')
        axes[1, 0].set_xlabel('')

        sns.boxplot(ax=axes[1, 1], data=df, x='algorithm', y='dist_3d', palette=palette, hue='algorithm', legend=False)
        axes[1, 1].set_title('3D Distance (Meters)', fontweight='bold')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].set_xlabel('')

        plt.tight_layout()
        out_img_box = f"graphs_comparison_{map_type.lower()}.png"
        plt.savefig(out_img_box, dpi=300, bbox_inches='tight')
        plt.close()

        # ==========================================
        # 3. GENERATE LINE GRAPH (Energy over Sequential Rows)
        # ==========================================
        plt.figure(figsize=(12, 6))
        
        algorithms = df['algorithm'].unique()
        max_rows = 0  # To keep track of the longest dataset for the x-axis ticks
        
        for algo in algorithms:
            # Filter the dataframe for this specific algorithm
            subset = df[df['algorithm'] == algo]
            num_rows = len(subset)
            
            if num_rows > max_rows:
                max_rows = num_rows
                
            # Create a sequential list (1, 2, 3...) based purely on row count
            x_values = range(1, num_rows + 1)
            
            plt.plot(
                x_values, 
                subset['energy'].values, 
                marker=markers[algo], 
                color=palette[algo], 
                label=algo, 
                linewidth=2, 
                markersize=8
            )
        
        # Formatting
        plt.title(f'Energy Consumption Over Sequential Runs - {map_type.capitalize()} Map', fontweight='bold', fontsize=14)
        plt.xlabel('Occurrence (Row Order)', fontweight='bold', fontsize=12)
        plt.ylabel('Energy (Joules)', fontweight='bold', fontsize=12)
        
        # Ensure the x-axis strictly shows integers up to the maximum number of occurrences
        if max_rows > 0:
            plt.xticks(range(1, max_rows + 1))
        
        # Clean up the legend
        plt.legend(title='', fontsize=11, loc='best')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        out_img_line = f"graphs/graphs_energy_over_runs_{map_type.lower()}.png"
        plt.savefig(out_img_line, dpi=300, bbox_inches='tight')
        print(f"✅ Graphs saved: {out_img_box} AND {out_img_line}")
        plt.close()

if __name__ == "__main__":
    generate_stats_and_graphs()