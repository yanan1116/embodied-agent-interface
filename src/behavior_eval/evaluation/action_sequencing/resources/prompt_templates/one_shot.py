# https://github.com/embodied-agent-interface/embodied-agent-interface/blob/main/src/behavior_eval/evaluation/action_sequencing/resources/prompt_templates/one_shot.py

prompt_template_onego = """

{problem_defination}

{data_format_instruction}
{action_format_instruction}

{action_explanations}

{special_attentions}
{safety_instruction}


{oneshot_example_head}

Please output the list of action commands (in the given format) so that after the robot executes the action commands sequentially, the current environment state will change to target environment state. Usually, the robot needs to execute multiple action commands consecutively to achieve final state. Please output multiple action commands rather than just one. Only output the list of action commands with nothing else.

Output:
{oneshot_example_output}

Your task:

Input:

instruction: {instruction}

initial environment state:
{init_state}

target environment state:
{target_state}

interactable objects:
{obj_list}

{feedback}

Please output the list of action commands (in the given format) so that after the robot executes the action commands sequentially, the current environment state will change to target environment state. Usually, the robot needs to execute multiple action commands consecutively to achieve final state. Please output multiple action commands rather than just one. Only output the list of action commands with nothing else.

Output:
"""


prompt_template_stepwise="""

{problem_defination}

{data_format_instruction}
{action_format_instruction}

{action_explanations}

{special_attentions}
{safety_instruction}

{oneshot_example_head}

At each step, you should generate just one action command with nothing else, which will be executed by the robot and the environment will be changed accordingly. 
for example like this:
{oneshot_example_output}


if you are provided with any feedback, suggestion or error hint, you need to comprehend and digest them and correct your last action command and return a revised command.
do not return repetitive action command; try to finish the task within 50 steps.
-----------------------------------
Your task:

Input:

instruction: {instruction}

initial environment state:
{init_state}

target environment state:
{target_state}

interactable objects:
{obj_list}

Your action command at this step:
"""



prompt_for_oracle_onego = """
Try to analyze why the agent's experimental action trajectory does not achieve the goal.

Here is the original prompt:
<<<< prompt start >>>>
{msg_prompt}
<<<< prompt end >>>>

Here is the experimental action trajectory from the agent:
<<<< trajectory start >>>>
{exp_trajectory}
<<<< trajectory end >>>>

Here is the ground-truth annotation action trajectory, which can successfully achieve the goal:
<<<< trajectory start >>>>
{annotation_trajectory}
<<<< trajectory end >>>>

Give your analysis about the failure reason and suggestions, which can be used by the agent for improvement next time:
(it would be better if your output is in markdown well-organized manner; there is no need to generate json parsable format)
"""



reflexion_prompt = """
your trajectory of action commands shown above does not achieve success. 
do discussion and reflection on why it caused failure. 
Do not return json parsable format. 
Just return a paragragh of natural language analysis. Be concise, logical, brief and to the point.
"""


stepwise_system_prompt = "For this task, please only output a parsable json string. Please start your answer with { and end your answer with }."

safety_instruction = """
9. If you have toggled on any object such as sink, microwave, during your execution, remember to toggle off it at last.
10. If you have open any object such as fridge, cabinet, door, window,  during your execution, remember to close it at last.
11. At last, the knife should be placed at the original place, such as on countertop."""


action_explanations = """
The action_name must be one of the following:
LEFT_GRASP # the robot grasps the object with its left hand, to execute the action, the robot's left hand must be empty, e.g. {'action': 'LEFT_GRASP', 'object': 'apple_0'}.
RIGHT_GRASP # the robot grasps the object with its right hand, to execute the action, the robot's right hand must be empty, e.g. {'action': 'RIGHT_GRASP', 'object': 'apple_0'}.
LEFT_PLACE_ONTOP # the robot places the object in its left hand on top of the target object and release the object in its left hand, e.g. {'action': 'LEFT_PLACE_ONTOP', 'object': 'table_1'}.
RIGHT_PLACE_ONTOP # the robot places the object in its right hand on top of the target object and release the object in its left hand, e.g. {'action': 'RIGHT_PLACE_ONTOP', 'object': 'table_1'}.
LEFT_PLACE_INSIDE # the robot places the object in its left hand inside the target object and release the object in its left hand, to execute the action, the robot's left hand must hold an object, and the target object can't be closed e.g. {'action': 'LEFT_PLACE_INSIDE', 'object': 'fridge_1'}.
RIGHT_PLACE_INSIDE # the robot places the object in its right hand inside the target object and release the object in its left hand, to execute the action, the robot's right hand must hold an object, and the target object can't be closed, e.g. {'action': 'RIGHT_PLACE_INSIDE', 'object': 'fridge_1'}.
RIGHT_RELEASE # the robot directly releases the object in its right hand, to execute the action, the robot's left hand must hold an object, e.g. {'action': 'RIGHT_RELEASE', 'object': 'apple_0'}.
LEFT_RELEASE # the robot directly releases the object in its left hand, to execute the action, the robot's right hand must hold an object, e.g. {'action': 'LEFT_RELEASE', 'object': 'apple_0'}.
OPEN # the robot opens the target object, to execute the action, the target object should be openable and closed, also, toggle off the target object first if want to open it, e.g. {'action': 'OPEN', 'object': 'fridge_1'}.
CLOSE # the robot closes the target object, to execute the action, the target object should be openable and open, e.g. {'action': 'CLOSE', 'object': 'fridge_1'}.
COOK # the robot cooks the target object, to execute the action, the target object should be put in a pan, e.g. {'action': 'COOK', 'object': 'apple_0'}.
CLEAN # the robot cleans the target object, to execute the action, the robot should have a cleaning tool such as rag, the cleaning tool should be soaked if possible, or the target object should be put into a toggled on cleaner like a sink or a dishwasher, e.g. {'action': 'CLEAN', 'object': 'window_0'}.
FREEZE # the robot freezes the target object e.g. {'action': 'FREEZE', 'object': 'apple_0'}.
UNFREEZE # the robot unfreezes the target object, e.g. {'action': 'UNFREEZE', 'object': 'apple_0'}.
SLICE # the robot slices the target object, to execute the action, the robot should have a knife in hand, e.g. {'action': 'SLICE', 'object': 'apple_0'}.
SOAK # the robot soaks the target object, to execute the action, the target object must be put in a toggled on sink, e.g. {'action': 'SOAK', 'object': 'rag_0'}.
DRY # the robot dries the target object, e.g. {'action': 'DRY', 'object': 'rag_0'}.
TOGGLE_ON # the robot toggles on the target object, to execute the action, the target object must be closed if the target object is openable and open e.g. {'action': 'TOGGLE_ON', 'object': 'light_0'}.
TOGGLE_OFF # the robot toggles off the target object, e.g. {'action': 'TOGGLE_OFF', 'object': 'light_0'}.
LEFT_PLACE_NEXTTO # the robot places the object in its left hand next to the target object and release the object in its left hand, e.g. {'action': 'LEFT_PLACE_NEXTTO', 'object': 'table_1'}.
RIGHT_PLACE_NEXTTO # the robot places the object in its right hand next to the target object and release the object in its right hand, e.g. {'action': 'RIGHT_PLACE_NEXTTO', 'object': 'table_1'}.
LEFT_TRANSFER_CONTENTS_INSIDE # the robot transfers the contents in the object in its left hand inside the target object, e.g. {'action': 'LEFT_TRANSFER_CONTENTS_INSIDE', 'object': 'bow_1'}.
RIGHT_TRANSFER_CONTENTS_INSIDE # the robot transfers the contents in the object in its right hand inside the target object, e.g. {'action': 'RIGHT_TRANSFER_CONTENTS_INSIDE', 'object': 'bow_1'}.
LEFT_TRANSFER_CONTENTS_ONTOP # the robot transfers the contents in the object in its left hand on top of the target object, e.g. {'action': 'LEFT_TRANSFER_CONTENTS_ONTOP', 'object': 'table_1'}.
RIGHT_TRANSFER_CONTENTS_ONTOP # the robot transfers the contents in the object in its right hand on top of the target object, e.g. {'action': 'RIGHT_TRANSFER_CONTENTS_ONTOP', 'object': 'table_1'}.
LEFT_PLACE_NEXTTO_ONTOP # the robot places the object in its left hand next to target object 1 and on top of the target object 2 and release the object in its left hand, e.g. {'action': 'LEFT_PLACE_NEXTTO_ONTOP', 'object': 'window_0, table_1'}.
RIGHT_PLACE_NEXTTO_ONTOP # the robot places the object in its right hand next to object 1 and on top of the target object 2 and release the object in its right hand, e.g. {'action': 'RIGHT_PLACE_NEXTTO_ONTOP', 'object': 'window_0, table_1'}.
LEFT_PLACE_UNDER # the robot places the object in its left hand under the target object and release the object in its left hand, e.g. {'action': 'LEFT_PLACE_UNDER', 'object': 'table_1'}.
RIGHT_PLACE_UNDER # the robot places the object in its right hand under the target object and release the object in its right hand, e.g. {'action': 'RIGHT_PLACE_UNDER', 'object': 'table_1'}.
DONE # the robot thinks the task has been completed.
"""

special_attentions = """
Please pay special attention:
1. The robot can only hold one object in each hand.
2. Action name must be one of the above action names, and the object name must be one of the object names listed in the interactable objects.
3. All PLACE actions will release the object in the robot's hand, you don't need to explicitly RELEASE the object after the PLACE action.
4. For LEFT_PLACE_NEXTTO_ONTOP and RIGHT_PLACE_NEXTTO_ONTOP, the action command are in the format of {'action': 'action_name', 'object': 'obj_name1, obj_name2'}
5. If you want to perform an action to an target object, you must make sure the target object is not inside a closed object.
6. For actions like OPEN, CLOSE, SLICE, COOK, CLEAN, SOAK, DRY, FREEZE, UNFREEZE, TOGGLE_ON, TOGGLE_OFF, at least one of the robot's hands must be empty, and the target object must have the corresponding property like they're openable, toggleable, etc.
7. For PLACE actions and RELEASE actions, the robot must hold an object in the corresponding hand.
8. Before slicing an object, the robot can only interact with the object (e.g. peach_0), after slicing the object, the robot can only interact with the sliced object (e.g. peach_0_part_0).
"""

problem_defination = """
Problem:
You are designing instructions for a household robot. 
The goal is to guide the robot to modify its environment from an initial state to a desired final state. 
The input will be the initial environment state, the target environment state, the objects you can interact with in the environment. 
The output should be a list of action commands so that after the robot executes the action commands sequentially, the environment will change from the initial state to the target state. 
"""

data_format_instruction = """
Data format: After # is the explanation.

Format of the states:
The environment state is a list starts with a uniary predicate or a binary prediate, followed by one or two obejcts.
You will be provided with multiple environment states as the initial state and the target state.
For example:
['inside', 'strawberry_0', 'fridge_97'] #strawberry_0 is inside fridge_97
['not', 'sliced', 'peach_0'] #peach_0 is not sliced
['ontop', 'jar_1', 'countertop_84'] #jar_1 is on top of countertop_84

Format of the interactable objects:
Interactable object will contain multiple lines, each line is a dictionary with the following format:
{
    "name": "object_name",
    "category": "object_category"
}
object_name is the name of the object, which you must use in the action command, object_category is the category of the object, which provides a hint for you in interpreting initial and goal condtions.
"""



# action_format_instruction
action_format_instruction_direct = """
Format of the action command:
Action command is a dictionary with the following format:
{
        "action": "action_name", 
        "object": "target_obj_name",
}

or 

{
        "action": "action_name", 
        "object": "target_obj_name1,target_obj_name2",
}

or 

{
        "action": "DONE", 
        "object": "",
}

"""

action_format_instruction_react = """
Format of the action command:
Action command is a dictionary with the following format:
{
        "action": "action_name", 
        "object": "target_obj_name",
        "rationale": "rationale",
}

or 

{
        "action": "action_name", 
        "object": "target_obj_name1,target_obj_name2",
        "rationale": "rationale",
}

or 

{
        "action": "DONE", 
        "object": "",
        "rationale": "",
}

"""




oneshot_example_head = """
Examples: after# is the explanation.

Example 1:
Input: 

instruction: Clean the stained bathtub and sink.

initial environment state:
['stained', 'sink_7']
['stained', 'bathtub_4']
['not', 'soaked', 'rag_0']
['onfloor', 'rag_0', 'room_floor_bathroom_0']
['inside', 'rag_0', 'cabinet_1']
['not', 'open', 'cabinet_1']

target environment state:
['not', 'stained', 'bathtub_4']
['not', 'stained', 'sink_7']
['and', 'soaked', 'rag_0', 'inside', 'rag_0', 'bucket_0']

interactable objects:
{'name': 'sink_7', 'category': 'sink.n.01'}
{'name': 'bathtub_4', 'category': 'bathtub.n.01'}
{'name': 'bucket_0', 'category': 'bucket.n.01'}
{'name': 'rag_0', 'category': 'rag.n.01'}
{'name': 'cabinet_1', 'category': 'cabinet.n.01'}
"""


# oneshot_example_output
oneshot_example_output_onego_direct = """
[
    {
        "action": "OPEN",
        "object": "cabinet_1"
    }, # you want to get the rag_0 from cabinet_1, should open it first
    {
        "action": "RIGHT_GRASP",
        "object": "rag_0"
    }, # you want to clean the sink_7 and bathtub_4, you found them stained, so you need to soak the rag_0 first
    {
        "action": "RIGHT_PLACE_INSIDE",
        "object": "sink_7"
    }, # to soak the rag_0, you need to place it inside the sink_7
    {
        "action": "TOGGLE_ON",
        "object": "sink_7"
    }, # to soak the rag_0, you need to toggle on the sink_7
    {
        "action": "SOAK",
        "object": "rag_0",
    }, # now you can soak the rag_0
    {
        "action": "TOGGLE_OFF",
        "object": "sink_7"
    }, # after soaking the rag_0, you need to toggle off the sink_7
    {
        "action": "LEFT_GRASP",
        "object": "rag_0"
    }, # now you can grasp soaked rag_0 to clean stain
    {
        "action": "CLEAN",
        "object": "sink_7"
    }, # now you clean the sink_7
    {
        "action": "CLEAN",
        "object": "bathtub_4"
    }, # now you clean the bathtub_4
    {
        "action": "LEFT_PLACE_INSIDE",
        "object": "bucket_0"
    }, # after cleaning the sink_7, you need to place the rag_0 inside the bucket_0
    {
        "action": "DONE",
        "object": ""
    }
]
"""

oneshot_example_output_onego_react = """
[
    {
        "rationale": "usually i need to grab the cleaning tools such as rag from the storage cabinet, so firstly open cabinet",
        "action": "OPEN",
        "object": "cabinet_1"
    }, 
    {
        "rationale": "if there is rag or other cleaning tools/objects in the cabinet, then i need to grasp it for later use.",
        "action": "RIGHT_GRASP",
        "object": "rag_0"
    }, 
    {
        "rationale": "before doing cleaning task, i need to soak the rag for preparation. so this step i need to place it inside the sink to soak the rag.",
        "action": "RIGHT_PLACE_INSIDE",
        "object": "sink_7"
    }, 
    {
        "rationale": "to soak the rag, i need to toggle on the sink",
        "action": "TOGGLE_ON",
        "object": "sink_7"
    }, 
    {
        "rationale": "the sink has been toggled on, so i can soak the rag now",
        "action": "SOAK",
        "object": "rag_0",
    }, 
    {
        "rationale": "after soaking the rag, i need to toggle off the sink as necessary step to avoid potential safety hazard",
        "action": "TOGGLE_OFF",
        "object": "sink_7"
    }, 
    {
        "rationale": "my left hand is idle, so now i can grasp soaked rag to clean stain",
        "action": "LEFT_GRASP",
        "object": "rag_0"
    }, 
    {
        "rationale": "now i can clean the sink, according to the instruction: Clean the stained bathtub and sink", 
        "action": "CLEAN",
        "object": "sink_7"
    }, 
    {
        "rationale": "in addition, now i can clean the bathtub, according to the instruction: Clean the stained bathtub and sink",
        "action": "CLEAN",
        "object": "bathtub_4"
    }, 
    {
        "rationale": "finally, after cleaning the sink, i need to place the rag in my left hand back inside the bucket.",
        "action": "LEFT_PLACE_INSIDE",
        "object": "bucket_0"
    },
    {
        "rationale": "",
        "action": "DONE",
        "object": ""
    }
]
"""


oneshot_example_output_stepwise_direct = """
[
    # at step 1
    {
        "action": "OPEN",
        "object": "cabinet_1"
    }

    # at step 2
    {
        "action": "RIGHT_GRASP",
        "object": "rag_0"
    }
    
    # at step 3
    {
        "action": "RIGHT_PLACE_INSIDE",
        "object": "sink_7"
    }
    
    # at step 4
    {
        "action": "TOGGLE_ON",
        "object": "sink_7"
    }
    
    # at step 5
    {
        "action": "SOAK",
        "object": "rag_0",
    }
    
    # at step 6
    {
        "action": "TOGGLE_OFF",
        "object": "sink_7"
    }
    
    # at step 7
    {
        "action": "LEFT_GRASP",
        "object": "rag_0"
    }
    
    # at step 8
    {
        "action": "CLEAN",
        "object": "sink_7"
    }
    
    # at step 9
    {
        "action": "CLEAN",
        "object": "bathtub_4"
    }
    
    # at step 10
    {
        "action": "LEFT_PLACE_INSIDE",
        "object": "bucket_0"
    } 

    # at step 11
    {
        "action": "CLOSE",
        "object": "cabinet_1"
    } 

    # at step 12
    {
        "action": "DONE",
        "object": ""
    } 
]
"""

oneshot_example_output_stepwise_react = """
[
    # at step 1
    {
        "rationale": "usually i need to grab the cleaning tools such as rag from the storage cabinet, so firstly open cabinet",
        "action": "OPEN",
        "object": "cabinet_1"
    }

    # at step 2
    {
        "rationale": "if there is rag or other cleaning tools/objects in the cabinet, then i need to grasp it for later use.",
        "action": "RIGHT_GRASP",
        "object": "rag_0"
    }
    
    # at step 3
    {
        "rationale": "before doing cleaning task, i need to soak the rag for preparation. so this step i need to place it inside the sink to soak the rag.",
        "action": "RIGHT_PLACE_INSIDE",
        "object": "sink_7"
    }
    
    # at step 4
    {
        "rationale": "to soak the rag, i need to toggle on the sink",
        "action": "TOGGLE_ON",
        "object": "sink_7"
    }
    
    # at step 5
    {
        "rationale": "the sink has been toggled on, so i can soak the rag now",
        "action": "SOAK",
        "object": "rag_0",
    }
    
    # at step 6
    {
        "rationale": "after soaking the rag, i need to toggle off the sink as necessary step to avoid potential safety hazard",
        "action": "TOGGLE_OFF",
        "object": "sink_7"
    }
    
    # at step 7
    {
        "rationale": "my left hand is idle, so now i can grasp soaked rag to clean stain",
        "action": "LEFT_GRASP",
        "object": "rag_0"
    }
    
    # at step 8
    {
        "rationale": "now i can clean the sink, according to the instruction: Clean the stained bathtub and sink",
        "action": "CLEAN",
        "object": "sink_7"
    }
    
    # at step 9
    {
        "rationale": "in addition, now i can clean the bathtub, according to the instruction: Clean the stained bathtub and sink",
        "action": "CLEAN",
        "object": "bathtub_4"
    }
    
    # at step 10
    {
        "rationale": "finally, after cleaning the sink, i need to place the rag in my left hand back inside the bucket.",
        "action": "LEFT_PLACE_INSIDE",
        "object": "bucket_0"
    } 

    # at step 11
    {
        "rationale": "i find that the cabinet is still open, to finish the task, the cabinet should be closed",
        "action": "CLOSE",
        "object": "cabinet_1"
    } 

    # at step 12
    {
        "rationale": "after reviewing all executed action commands, i think the task has been finished.",
        "action": "DONE",
        "object": ""
    } 
]
"""

oneshot_example_output_stepwise_rej = """
(note that in the rationale, you should include INCORRECT and CORRECT actions for each step)

[
    # at step 1
    {
        "rationale": "
                        INCORRECT ACTIONS: 
                            ignore checking if cabinet is within the available interactable objects in the scene;
                            directly grab the clearning tools;
                            ignore to check if the cabinet is open or close.
                        CORRECT ACTION: 
                            check if the cabinet is open, otherwise open it at first;
                            usually i need to grab the cleaning tools such as rag from the storage cabinet, so firstly open cabinet.
                    ",
        "action": "OPEN",
        "object": "cabinet_1"
    }

    # at step 2
    {
        "rationale": "
                        INCORRECT ACTIONS:
                            close the cabinet;
                            toggle on sink;
                            toggle off sink;
                            soak rag;
                            ignore the rag in the cabinet.
                        CORRECT ACTION: 
                            if there is rag or other cleaning tools/objects in the cabinet, then i need to grasp it for later use.
                    ",
        "action": "RIGHT_GRASP",
        "object": "rag_0"
    }
    
    # at step 3
    {
        "rationale": "
                        INCORRECT ACTIONS:
                           soak rag since sink has not been toggled on;
                           toggle off sink;
                           clean sink since rag has not been soakedd;
                           clean bathtub
                        CORRECT ACTION:
                            before doing cleaning task, i need to soak the rag for preparation. so this step i need to place it inside the sink to soak the rag.
                     ",
        "action": "RIGHT_PLACE_INSIDE",
        "object": "sink_7"
    }
    
    # at step 4
    {
        "rationale": "
                        INCORRECT ACTIONS:
                            ignore checking the status of the sink - whether it is toggled on or off;
                            toggle off sink;
                            clean bathtub and sink since the rag has not been ready for use
                        CORRECT ACTION:
                            to soak the rag, i need to toggle on the sink.
                    ",
        "action": "TOGGLE_ON",
        "object": "sink_7"
    }
    
    # at step 5
    {
        "rationale": "
                        INCORRECT ACTIONS:
                            toggle on or off sink as sink has alread been toggled on;
                            clean bathtub and sink since the rag has not been ready for use;
                            open or close the cabinet;
                            place rag in any bucket
                        CORRECT ACTION:
                            the sink has been toggled on, so i can soak the rag now
                     ",
        "action": "SOAK",
        "object": "rag_0",
    }
    
    # at step 6
    {
        "rationale": "
                        INCORRECT ACTIONS:
                            open or close the cabinet;
                            toggle on the sink because the sink should be off after soaking the rag;
                            place rag in any bucket
                        CORRECT ACTION:
                            after soaking the rag, i need to toggle off the sink as necessary step to avoid potential safety hazard
                      ",
        "action": "TOGGLE_OFF",
        "object": "sink_7"
    }
    
    # at step 7
    {
        "rationale": "
                        INCORRECT ACTIONS:
                            open or close the cabinet;
                            place rag in any bucket;
                            toggle on or off sink as sink has alread been toggled off;
                            clean bathtub and sink since rag should be grasped before cleaning
                        CORRECT ACTION:
                            my left hand is idle, so now i can grasp soaked rag to clean stain
                    ",
        "action": "LEFT_GRASP",
        "object": "rag_0"
    }
    
    # at step 8
    {
        "rationale": "
                        INCORRECT ACTIONS:
                            grasp rag as it is already in the hand;
                            open or close the cabinet;
                            place rag in any bucket;
                            toggle on or off sink as sink has alread been toggled off
                        CORRECT ACTION:
                            now i can clean the sink, according to the instruction: Clean the stained bathtub and sink",
        "action": "CLEAN",
        "object": "sink_7"
    }
    
    # at step 9
    {
        "rationale": "
                        INCORRECT ACTIONS:
                            grasp rag as it is already in the hand;
                            open or close the cabinet;
                            place rag in any bucket;
                            toggle on or off sink as sink has alread been toggled off;
                            clean the sink as it has already been cleaned in last step
                        CORRECT ACTION:
                            in addition, now i can clean the bathtub, according to the instruction: Clean the stained bathtub and sink
                     ",
        "action": "CLEAN",
        "object": "bathtub_4"
    }
    
    # at step 10
    {
        "rationale": "
                        INCORRECT ACTIONS:
                            open or close the cabinet;
                            grasp rag as it is already in the hand;
                            toggle on or off sink as sink has alread been toggled off;
                            clean the sink or bathtub as they have already been cleaned in last step                            
                        CORRECT ACTION:
                            finally, after cleaning is done, i need to place the rag in my left hand back inside the bucket.
                     ",
        "action": "LEFT_PLACE_INSIDE",
        "object": "bucket_0"
    } 

    # at step 11
    {
        "rationale": "
                        INCORRECT ACTIONS:
                            open the cabinet;
                        CORRECT ACTION:
                            i find that the cabinet is still open, to finish the task, the cabinet should be closed",
        "action": "CLOSE",
        "object": "cabinet_1"
    } 

    # at step 12
    {
        "rationale": "after reviewing all executed action commands, i think the task has been finished.",
        "action": "DONE",
        "object": ""
    } 
]
"""