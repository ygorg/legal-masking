import os
import json
import logging
from glob import glob
from collections import defaultdict

import pandas as pd
from scipy.stats import ttest_ind


def arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Gather and display results")
    parser.add_argument('-d', '--logdir', default='logs', help="Directory to search for results (default: ./logs)")
    parser.add_argument('-l', '--to-latex', action="store_true", help="Print result table as latex.")
    parser.add_argument('-c', '--to-csv', action="store_true", help="Print result table as csv.")
    parser.add_argument('--round', type=int, default="2", help="Rounding precision (default: 2).")
    parser.add_argument('-r', '--raw', action="store_true", help="Print all results as csv (will not print result table).")
    parser.add_argument('--sep', type=str, default='.', help="Separator for decimal in csv (default: '.').")
    return parser.parse_args()


args = arguments()

logging.basicConfig(level=logging.WARNING)

# For every task/model/seed get the metrics
root_path = args.logdir
all_scores = defaultdict(lambda: {"macro-f1": [], "micro-f1": []})
for task_path in glob(os.path.join(root_path, '*')):
    task = os.path.split(task_path)[-1]
    for model_path in glob(os.path.join(task_path, '*')):
        model_name = os.path.split(model_path)[-1]
        for seed_path in glob(os.path.join(model_path, '*')):
            seed = os.path.split(seed_path)[-1]
            score_fn = os.path.join(seed_path, "predict_results.json")
            if not os.path.isfile(score_fn):
                continue
            with open(score_fn) as f:
                data = json.load(f)
            all_scores[(task, model_name, seed)]["macro-f1"] = round(data.get('predict_macro-f1', 0) * 100, args.round)
            all_scores[(task, model_name, seed)]["micro-f1"] = round(data.get('predict_micro-f1', 0) * 100, args.round)


if not all_scores:
    logging.error('Could not find scores.')
    logging.error('Directory structure should be DIR/task/model/seed/predict_results.json')
    exit()

# Using pandas create a nice result table or export the results to excel
df = pd.DataFrame(all_scores).T
df.index.names = ["task", "path", "seed"]
df = df.reset_index()
# df = df * 100

# extract parameters from model_path
df["model"] = df["path"].map(lambda x: x.split('-')[0])  # should be "bert" or "legalbert"
df["weight"] = df["path"].map(lambda x: x.split('-')[-2])  # should be "tfidf", "meta" or "cft"
df["strat"] = df["path"].map(lambda x: x.split('-')[-1])  # should be "topn" or "rw"

if args.raw:
    # Save to csv
    print(df.to_csv(decimal=args.sep, index=False))
    # df.to_excel('results.xlsx', index=False)
    exit()


def signif(sdf):
    global df
    # `sdf` contains all the seeds results for a particular model
    # we search in the global df (that contains everything)
    #  the baseline that corresponds to the current model
    baseline = df.loc[df['model'] == sdf['model'].iloc[0]]\
                 .loc[df['task'] == sdf['task'].iloc[0]]\
                 .loc[df['weight'] == 'baseline']\
                 .loc[df['strat'] == 'cpt']

    # The line that we will return
    new_values = {}
    # Count the number of seeds (number of lines we got)
    new_values['nseed'] = len(sdf)

    # For each metric
    for m in ['micro-f1', 'macro-f1']:
        # Get the score
        baseline_res = baseline[m]  # 1 value
        model_res = sdf[m]  # `nseed` values

        # Compute significance test
        t_statistic, p_value = ttest_ind(baseline_res, model_res)
        if p_value < 0.01:
            interpretation = '++'
        elif p_value < 0.05:
            interpretation = '+'
        else:
            interpretation = ''

        # Display the significance along the score
        new_values[m] = f'{sdf[m].mean():.2f} {interpretation}'

    # Return the result of the aggregation of the lines of `sdf`
    return pd.Series(new_values)


# Groupby aggregate all the seeds in only one row :
# - compute mean of each metric
# - compute significance
metrics = df.groupby(['model', 'weight', 'strat', 'task'])\
            .apply(signif)

# Warn if number of seed is inconsistent
most_common_nb_seeds = metrics['nseed'].mode()[0]  # .mode() returns the values ordered by frequency
missing_seed = metrics[metrics['nseed'] != most_common_nb_seeds].index
for i in missing_seed:
    logging.warning(f"Beware ! {i} was computed on {int(metrics.loc[i]['nseed'])} seeds and not {most_common_nb_seeds} as the rest.")

# We warned no need to keep the number of seeds anymore
metrics = metrics.drop(columns=('nseed'))

# Group by task
table = metrics.unstack(3)
# Nicely display per task
table.columns = table.columns.swaplevel(0, 1)
table.sort_index(axis=1, level=0, inplace=True)
# table = table.round(args.round)

if args.to_latex:
    print(table.to_latex())
elif args.to_csv:
    print(table.to_csv(decimal=args.sep))
else:
    print(table)
