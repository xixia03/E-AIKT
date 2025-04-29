import torch
from collections import defaultdict

#merging multiple samples (a batch of data) into a single tensor for training or inference
def custom_collate_fn(batch):
    num_problems=len(batch)
    question_ids, question_masks = [], []
    all_desc_ids, all_desc_masks, desc_lengths = [], [], []
    all_kc_ids, all_kc_masks, kc_lengths = [], [], []
    all_kc_quest_pairs = []
    all_desc_quest_pairs = []  
    all_kc_cluster_ids, all_desc_cluster_ids= [],[]
    # cluster_kc_quest_pairs
    cluster_kc_quest_pairs = []
    cluster_desc_quest_pairs = []

    #cluster_to_kcs: defaultdict(<class 'list'>, {'0': [0, 3, 4, 7], '1': [1, 5, 6, 8], '2': [2]}) 
    cluster_to_kcs = defaultdict(list)
    #cluster_to_questions_kc: defaultdict(<class 'set'>, {'0': {0, 1, 2}, '1': {0, 1, 2}, '2': {0}})
    cluster_to_questions_kc = defaultdict(set)
    cluster_to_descs = defaultdict(list)
    cluster_to_questions_desc = defaultdict(set)

    num_prev_descs = 0
    num_prev_kcs = 0


    for i,item in enumerate(batch):
        question_ids.append(item['question_ids'])
        question_masks.append(item['question_mask'])
        all_desc_ids.append(item['desc_ids'])
        all_desc_masks.append(item['desc_mask'])
        cur_num_descs = item['desc_ids'].size(0)
        desc_lengths.append(cur_num_descs)
    
        all_kc_ids.append(item['kc_ids'])
        all_kc_masks.append(item['kc_mask'])
        cur_num_kcs = item['kc_ids'].size(0)
        kc_lengths.append(cur_num_kcs)
        all_kc_cluster_ids.append(item['kc_cluster_ids'])
        all_desc_cluster_ids.append(item['desc_cluster_ids'])

        kc_quest_pairs = torch.ones((cur_num_kcs, 2)) * i
        kc_quest_pairs[:, 0] = torch.arange(cur_num_kcs)  # all kcs point to question i
        kc_quest_pairs[:, 0] += num_prev_kcs  # kcs shifted correctly
        kc_quest_pairs = kc_quest_pairs.long()
        all_kc_quest_pairs.append(kc_quest_pairs)
        assert kc_quest_pairs[:, 0].max() <= sum(kc_lengths), "kc index out of bounds"
        assert kc_quest_pairs[:, 1].max() <= num_problems, "question index out of bounds"
    
        desc_quest_pairs = torch.ones((cur_num_descs, 2)) * i
        desc_quest_pairs[:, 0] = torch.arange(cur_num_descs)  # all descs point to question i
        desc_quest_pairs[:, 0] += num_prev_descs  # descriptions shifted correctly
        desc_quest_pairs = desc_quest_pairs.long()
        all_desc_quest_pairs.append(desc_quest_pairs)
        assert desc_quest_pairs[:, 0].max() <= sum(desc_lengths), "desc index out of bounds"
        assert desc_quest_pairs[:, 1].max() <= num_problems, "question index out of bounds"

        for kc_idx, cluster_id in enumerate(item['kc_cluster_ids']):
            cluster_to_kcs[cluster_id.item()].append(num_prev_kcs + kc_idx)
            cluster_to_questions_kc[cluster_id.item()].add(i)
        
        for desc_idx, cluster_id in enumerate(item['desc_cluster_ids']):
            cluster_to_descs[cluster_id.item()].append(num_prev_descs + desc_idx)
            cluster_to_questions_desc[cluster_id.item()].add(i)

        num_prev_kcs += cur_num_kcs
        num_prev_descs += cur_num_descs
    
    for cluster_id, kcs in cluster_to_kcs.items():
        questions = list(cluster_to_questions_kc[cluster_id])  # Convert set to list to index it
        for kc in kcs:
            for question in questions:
                cluster_kc_quest_pairs.append([kc, question])
    
    for cluster_id, descs in cluster_to_descs.items():
        questions = list(cluster_to_questions_desc[cluster_id])
        for desc in descs:
            for question in questions:
                cluster_desc_quest_pairs.append([desc, question])
                
    question_ids = torch.stack(question_ids)
    question_masks = torch.stack(question_masks)    
    all_desc_ids = torch.cat(all_desc_ids)
    all_desc_masks = torch.cat(all_desc_masks)
    all_kc_ids = torch.cat(all_kc_ids)
    all_kc_masks = torch.cat(all_kc_masks)
    all_kc_quest_pairs = torch.cat(all_kc_quest_pairs)
    all_desc_quest_pairs = torch.cat(all_desc_quest_pairs)
    all_cluster_kc_quest_pairs = torch.tensor(cluster_kc_quest_pairs, dtype=torch.long)
    all_cluster_desc_quest_pairs = torch.tensor(cluster_desc_quest_pairs, dtype=torch.long)
    desc_lengths = torch.tensor(desc_lengths)
    kc_lengths = torch.tensor(kc_lengths)

    return {
        'question_ids': question_ids,
        'question_mask': question_masks,
        'desc_ids': all_desc_ids,
        'desc_mask': all_desc_masks,
        'desc_lengths': desc_lengths,
        'kc_ids': all_kc_ids,
        'kc_mask': all_kc_masks,
        'kc_lengths': kc_lengths,
        'kc_quest_pairs': all_kc_quest_pairs,
        'desc_quest_pairs': all_desc_quest_pairs,
        'cluster_kc_quest_pairs': all_cluster_kc_quest_pairs,
        'cluster_desc_quest_pairs': all_cluster_desc_quest_pairs
    }
