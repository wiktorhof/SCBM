import wandb
import pandas as pd
import numpy as np
from tqdm import tqdm

# Initialize API and get runs
api = wandb.Api()
wandb_entity='wiktorh'
project_name='PSCBM'
runs = api.runs(f"{wandb_entity}/{project_name}")  # Replace with your project path

#Interesting tresholds for CUB - hardcoded
auc_tresholds = list(range(1,10,1))
auc_tresholds.extend(range(10,112,10))


all_runs_results = []

for run in tqdm(runs):
    try:
        # Get validation accuracy history with steps
        history = list(run.scan_history())
        results = {
            'run_id': run.id,
            'run_name': run.name,
            'model': run.config['model']['model'],
            'concept_learning': run.config['model']['concept_learning'],
            'cov_type': run.config['model'].get('cov_type', None),
            'dataset': run.config['data']['dataset'],
            }
        if not history:
            continue
            
        # Convert to DataFrame and clean data
        df = pd.DataFrame(history)
        selected_cols = [col for col in df.columns if 'intervention' in col and ('accuracy' in col or 'loss' in col)]
        for col in selected_cols:
            int_curve = df.loc[:,[col, 'intervention/num_concepts_intervened']].dropna().sort_values('intervention/num_concepts_intervened')
            value_at_0 = int_curve[col].iloc[0]
            # print(int_curve)
            int_curve.loc[:, col] -= value_at_0
            # print(int_curve)
            for treshold in auc_tresholds:
                sub_curve = int_curve.where(int_curve['intervention/num_concepts_intervened'] <= treshold).dropna().sort_values('intervention/num_concepts_intervened')
                # print(sub_curve)
                abs_improvement = sub_curve[col].iloc[-1]
                auc = np.trapz(sub_curve[col], sub_curve['intervention/num_concepts_intervened'])
                # print(auc)
                results[f'{col}/auc/{treshold}'] = auc
                results[f'{col}/improvement/{treshold}'] = abs_improvement

            auc_total = np.trapz(int_curve[col], int_curve['intervention/num_concepts_intervened'])
            improvement_total = int_curve[col].iloc[-1]
            results[f'{col}/total_auc'] = auc_total
            results[f'{col}/total_improvement'] = improvement_total

        # print(results)
        all_runs_results.append(results)
        
    except Exception as e:
        print(f"Error processing run {run.id}: {e}")
        continue

all_results = pd.DataFrame(all_runs_results)
print(all_results)

with wandb.init(entity=wandb_entity, project=project_name, mode='online', name='interventions_comparison_table'):
    final_table = wandb.Table(columns=all_results.columns, data=all_results.data, log_mode="IMMUTABLE")
    run.log({"table": final_table})

print(f"Processed {len(results)} runs successfully")
