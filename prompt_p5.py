"""
Pretraining Tasks -- 5 Prompt Families (1, 2, 3, 4, 5)
"""

# =====================================================
# Task Subgroup 1 -- Sequential item -- 10 Prompts
# =====================================================

task_subgroup_1 = []

template = {}
template[
    "source"
] = "User_{} has purchased : {} , predict next possible item to be bought by the user ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-1"

task_subgroup_1.append(template)

template = {}
template[
    "source"
] = "I find the purchase list of user_{} : {} , I wonder what other items does the user need . Can you help me decide ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-2"

task_subgroup_1.append(template)

template = {}
template[
    "source"
] = "Here is the purchase list of user_{} : {} , try to recommend another item to the user ."
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-3"

task_subgroup_1.append(template)

template = {}
template[
    "source"
] = "According to what items user_{} has purchased : {} , Can you recommend another item to the user ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-4"

task_subgroup_1.append(template)

template = {}
template[
    "source"
] = "The user_{} has bought items : {} , What else do you think is necessary for the user ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-5"

task_subgroup_1.append(template)

template = {}
template[
    "source"
] = "Here is the item purchase history of user_{} : {} , What to recommend next for the user ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-6"

task_subgroup_1.append(template)

template = {}
template["source"] = "What would user_{} be likely to purchase next after buying : {} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-7"

task_subgroup_1.append(template)

template = {}
template[
    "source"
] = "By analyzing the user_{} 's purchase of : {}, what is the next item expected to be bought ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-8"

task_subgroup_1.append(template)

template = {}
template[
    "source"
] = "Can you recommend the next item for user_{} , given the user's purchase of : {} ?"
template["target"] = "{}"
template["input_first"] = "user"
template["task"] = "sequential"
template["id"] = "1-9"

task_subgroup_1.append(template)

template = {}
template[
    "source"
] = "After buying items : {} , what is the next item that could be recommended for user_{} ?"
template["target"] = "{}"
template["input_first"] = ""
template["task"] = "sequential"
template["id"] = "1-10"

task_subgroup_1.append(template)
