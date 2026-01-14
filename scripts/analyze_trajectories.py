"""
Analyze trajectory files and save statistics to CSV.
Counts message kinds and extracts final_message for each trajectory.
"""

import json
import os
import re
from pathlib import Path
import pandas as pd


def analyze_trajectory(filepath: Path) -> dict:
    """Analyze a single trajectory JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Count message kinds
    messages = data.get('messages', [])
    kind_counts = {}
    for msg in messages:
        kind = msg.get('kind', 'Unknown')
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    
    # Extract final message (truncate to avoid huge cells)
    final_message = data.get('final_message', '')
    if len(final_message) > 500:
        final_message = final_message[:500] + '...'
    
    # Extract reward
    reward_dict = data.get('reward_dict', {})
    total_reward = sum(reward_dict.values()) if reward_dict else 0.0
    
    return {
        'kind_counts': kind_counts,
        'final_message': final_message,
        'total_reward': total_reward,
        'num_messages': len(messages),
    }


def main():
    base_dir = Path('/home/ubuntu/agentic-code-search-oss/ckpts/Qwen-Qwen3-4B/trajectories')
    
    rows = []
    all_kinds = set()
    
    # Find all trajectory files
    for step_dir in sorted(base_dir.iterdir()):
        if not step_dir.is_dir() or not step_dir.name.startswith('step_'):
            continue
        
        # Extract step number
        step_match = re.match(r'step_(\d+)', step_dir.name)
        step_num = int(step_match.group(1)) if step_match else 0
        
        # Process train and eval subdirs
        for phase_dir in step_dir.iterdir():
            if not phase_dir.is_dir():
                continue
            phase = phase_dir.name  # 'train' or 'eval'
            
            for json_file in phase_dir.glob('*.json'):
                # ID is filename without .json
                file_id = json_file.stem
                
                try:
                    result = analyze_trajectory(json_file)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue
                
                # Collect all message kinds seen
                all_kinds.update(result['kind_counts'].keys())
                
                row = {
                    'id': file_id,
                    'step': step_num,
                    'phase': phase,
                    'num_messages': result['num_messages'],
                    'total_reward': result['total_reward'],
                    'final_message': result['final_message'],
                    **{f'count_{k}': result['kind_counts'].get(k, 0) for k in result['kind_counts']}
                }
                rows.append(row)
    
    if not rows:
        print("No trajectory files found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Ensure all kind columns exist (fill missing with 0)
    kind_columns = [f'count_{k}' for k in sorted(all_kinds)]
    for col in kind_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns
    base_cols = ['id', 'step', 'phase', 'num_messages', 'total_reward']
    kind_cols = sorted([c for c in df.columns if c.startswith('count_')])
    other_cols = ['final_message']
    df = df[base_cols + kind_cols + other_cols]
    
    # Fill NaN with 0 for count columns
    for col in kind_cols:
        df[col] = df[col].fillna(0).astype(int)
    
    # Sort by step and id
    df = df.sort_values(['step', 'id']).reset_index(drop=True)
    
    # Save to CSV
    output_path = base_dir.parent / 'trajectory_analysis.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved analysis to: {output_path}")
    
    # Print summary
    print(f"\nTotal trajectories analyzed: {len(df)}")
    print(f"Steps covered: {df['step'].min()} to {df['step'].max()}")
    print(f"\nMessage kind columns: {kind_cols}")
    print(f"\nMessage kind totals:")
    for col in kind_cols:
        print(f"  {col}: {df[col].sum()}")
    print(f"\nAverage reward: {df['total_reward'].mean():.4f}")
    print(f"\nSample rows:")
    print(df[base_cols + kind_cols].head(10).to_string())


if __name__ == '__main__':
    main()

