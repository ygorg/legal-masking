import os
import datasets

for task in ['case_hold', 'ledgar', 'eurlex', 'ecthr_b',
             'ecthr_a', 'scotus', 'unfair_tos']:
    print(f'-- {task} -- ')
    datasets.load_dataset('lex_glue', name=task)\
            .save_to_disk(os.path.join('data', task))
