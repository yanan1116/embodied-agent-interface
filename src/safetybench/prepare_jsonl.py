

import json,sys,datasets,random,traceback
from behavior_eval.evaluation.action_sequencing.action_sequence_evaluator import ActionSequenceEvaluator
import ast

infos = []
with open("src/safetybench/VestaBench-B50_.jsonl", "r") as f:
    for line in f:
        ii = json.loads(line) 
        if len(ii['trajectory']) <= 10:
            ii['complexity'] = 'low'
        elif len(ii['trajectory']) > 10 and len(ii['trajectory']) <= 20:
            ii['complexity'] = 'medium'
        else:
            ii['complexity'] = 'high'
        
        demo_name = ii['task_id']

        ase = ActionSequenceEvaluator(demo_name=demo_name)
        success_goals_raw = ase.get_target_state()

        print(ii)
        print(success_goals_raw)
        ii['success_goals'] = [ast.literal_eval(line) for line in success_goals_raw.strip().split('\n')]
        print(ii['success_goals'])

        infos.append(ii)
        ase.transition_model.env.close()
        
with open("src/safetybench/VestaBench-B50.jsonl", "w") as f:
    for ii in infos:
        f.write(json.dumps(ii) + "\n")

categories = set()
task_category_dic = {}
with open("src/safetybench/VestaBench-B50.jsonl", "r") as f: 
    for line in f:
        dd = json.loads(line)  
        print(dd['risk_category'])
        categories.update(dd['risk_category'])
        task_category_dic[dd['task_id']] = dd['risk_category']
# ds = datasets.load_dataset('Inevitablevalor/EmbodiedAgentInterface')
# task_traj_dic = {ii['task_id']:eval(ii['action_trajectory']) for ii in ds}




