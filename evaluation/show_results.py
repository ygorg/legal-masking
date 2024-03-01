import os
import json
import logging
from glob import glob
from collections import defaultdict

import pandas as pd


def arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Gather and display results")
    parser.add_argument('-d', '--logdir', default='logs', help="Directory to search for results (default: ./logs)")
    parser.add_argument('-l', '--to-latex', action="store_true", help="Print result table as latex.")
    parser.add_argument('-c', '--to-csv', action="store_true", help="Print result table as csv.")
    parser.add_argument('--round', type=int, default="2", help="Rounding precision (default: 2).")
    parser.add_argument('-r', '--raw', action="store_true", help="Print all results as csv (will not print result table).")
    parser.add_argument('--sep', type=str, default='.', help="Separator for decimal in csv (default: '.').")
    parser.add_argument('--std', action="store_true", help="Compute std as well as mean.")
    return parser.parse_args()


args = arguments()

logging.basicConfig(level=logging.WARNING)

# For every task/model/seed get the metrics
root_path = args.logdir
all_scores = defaultdict(lambda: {"macro-f1": [], "micro-f1": []})
for task_path in glob(os.path.join(root_path, '*')):
    task = os.path.split(task_path)[-1]
    for model_path in glob(os.path.join(task_path, '*')):
        model = os.path.split(model_path)[-1]
        model = model.replace('legal-bert', 'legal_bert').replace('top-n', 'topn').replace('random-weighted', 'rand').replace('-base-uncased', '_base_uncased')
        model = model.split('-')
        if len(model) == 1:
            model += ['', '']
        if len(model) == 2:
            if model[1] in ('rand', 'topn'):
                model = [model[0], '', model[1]]
            else:
                model += ['']
        model = tuple(model)
        for seed_path in glob(os.path.join(model_path, '*')):
            seed = os.path.split(seed_path)[-1]
            score_fn = os.path.join(seed_path, "predict_results.json")
            if not os.path.isfile(score_fn):
                continue
            with open(score_fn) as f:
                data = json.load(f)
                all_scores[(task, *model, seed)]["macro-f1"] = data["predict_macro-f1"]
                all_scores[(task, *model, seed)]["micro-f1"] = data["predict_micro-f1"]

if not all_scores:
    logging.error('Could not find scores.')
    logging.error('Directory structure should be DIR/task/model/seed/predict_results.json')
    exit()

# Using pandas create a nice result table or export the results to excel
df = pd.DataFrame(all_scores).T
df = df * 100
df.index.names = ["task", "model", "weight", "strat", "seed"]
df = df.reset_index()

if args.raw:
    print(df.to_csv(decimal=args.sep, index=False))
    # df.to_excel('results.xlsx', index=False)
    exit()


def confidence(vals):
    from numpy import sqrt
    from scipy import stats

    return stats.norm.interval(
        0.95,
        loc=vals.mean(),
        scale=vals.std() / sqrt(len(vals))
    )


agg = {
    'seed': ['count'],
    'macro-f1': ['mean'],
    'micro-f1': ['mean'],
}
if args.std:
    agg['macro-f1'].append('std')
    agg['micro-f1'].append('std')


metrics = df.groupby(['model', 'weight', 'strat', 'task'])\
            .agg(agg)

# Warn if number of seed is inconsistent
most_common_nb_seeds = metrics[('seed', 'count')].mode()[0]
missing_seed = metrics[metrics[('seed', 'count')] != most_common_nb_seeds].index
for i in missing_seed:
    logging.warning(f"Beware ! {i} was computed on {int(metrics.loc[i]['seed'].iloc[0])} seeds and not {most_common_nb_seeds} as the rest.")

metrics = metrics.drop(columns=('seed', 'count'))

# Group by task
table = metrics.unstack(3)
# Nicely display per task
table.columns = table.columns.swaplevel(0, 2)
table.sort_index(axis=1, level=0, inplace=True)
table = table.round(args.round)

if args.to_latex:
    print(table.to_latex())
elif args.to_csv:
    print(table.to_csv(decimal=args.sep))
else:
    print(table)