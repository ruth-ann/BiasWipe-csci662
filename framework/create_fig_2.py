import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_evaluation_file(filepath):
    """
    Parse an evaluation file and extract FPR for each entity term.
    
    Returns:
        dict: {entity_term: fpr_value}
    """
    results = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by entity terms
    entity_blocks = re.split(r'ENTITY TERM:', content)
    
    for block in entity_blocks[1:]:  # Skip first empty split
        lines = block.strip().split('\n')
        entity_term = lines[0].strip()
        
        # Extract FPR
        for line in lines:
            if 'False Positive Rate:' in line:
                fpr_str = line.split(':')[1].strip().rstrip('%')
                fpr_value = float(fpr_str) / 100.0
                results[entity_term] = fpr_value
                break
    
    return results

def load_all_models(file_paths, model_names):
    """
    Load FPR data from multiple evaluation files.
    
    Args:
        file_paths: list of paths to evaluation files
        model_names: list of names for each model
    
    Returns:
        dict: {model_name: {entity: fpr}}
    """
    all_data = {}
    
    for path, name in zip(file_paths, model_names):
        all_data[name] = parse_evaluation_file(path)
    
    return all_data

def plot_fpr_comparison(all_data, output_path=None):
    """
    Create a grouped bar chart comparing FPR across models and entities.
    
    Args:
        all_data: dict of {model_name: {entity: fpr}}
        output_path: optional path to save the figure
    """
    # Get all unique entities and sort them by maximum FPR
    all_entities = set()
    for model_data in all_data.values():
        all_entities.update(model_data.keys())
    
    # Calculate max FPR for each entity across all models
    entity_max_fpr = {}
    for entity in all_entities:
        max_fpr = max(all_data[model].get(entity, 0) for model in all_data)
        entity_max_fpr[entity] = max_fpr
    
    # Sort entities by max FPR
    entities = sorted(all_entities, key=lambda x: entity_max_fpr[x])
    
    # Prepare data
    model_names = list(all_data.keys())
    n_models = len(model_names)
    n_entities = len(entities)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Define colors
    colors = ['#5B8FF9', '#E8684A', '#F6BD16', '#5AD8A6']
    
    # Calculate bar positions
    bar_width = 0.2
    x = np.arange(n_entities)
    
    # Plot bars for each model
    for i, model_name in enumerate(model_names):
        fpr_values = [all_data[model_name].get(entity, 0) for entity in entities]
        offset = (i - n_models/2 + 0.5) * bar_width
        ax.bar(x + offset, fpr_values, bar_width, label=model_name, color=colors[i])
    
    # Customize plot
    ax.set_xlabel('Keyword', fontsize=12)
    ax.set_ylabel('FPR', fontsize=12)
    ax.set_title('Figure 2: False positive rate for all entities for BERT model.', 
                 fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(entities, rotation=45, ha='right')
    ax.legend(loc='upper left', frameon=False)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define your evaluation file paths
    file_paths = [
        '/home/exouser/data/outputs/evaluations/old_bert_bias_eval.txt',
        '/home/exouser/data/outputs/evaluations/old_bert_gay_100_bias_eval.txt',
        '/home/exouser/data/outputs/evaluations/old_bert_gay_homosexual_100_bias_eval.txt',
        '/home/exouser/data/outputs/evaluations/old_bert_gay_homosexual_lesbian_100_bias_eval.txt'
    ]
    
    # Define model names
    model_names = ['BERT', 'Unlearn 1', 'Unlearn 2', 'Unlearn 3']
    
    # Load data
    all_data = load_all_models(file_paths, model_names)
    
    # Create visualization
    plot_fpr_comparison(all_data, output_path='fpr_comparison.png')
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for model_name, data in all_data.items():
        print(f"\n{model_name}:")
        sorted_entities = sorted(data.items(), key=lambda x: x[1], reverse=True)
        print(f"  Highest FPR: {sorted_entities[0][0]} ({sorted_entities[0][1]:.2%})")
        print(f"  Lowest FPR: {sorted_entities[-1][0]} ({sorted_entities[-1][1]:.2%})")
        print(f"  Average FPR: {np.mean(list(data.values())):.2%}")