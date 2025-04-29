import json
import torch
from torch.utils.data import Dataset

class MathDataset(Dataset):
    def __init__(self, tokenizer, args):
        # Load both dataset and cluster files: kc_annotations_data.json and kc_clusters_hdbscan.json
        with open(args.json_file_dataset, 'r') as f:
            self.data = json.load(f)
        with open(args.json_file_cluster_kc, 'r') as f:
            self.cluster_to_kcs = json.load(f)
        with open(args.json_file_cluster_desc, 'r') as f:
            self.cluster_to_descs = json.load(f)

        self.kc_to_cluster = self.__get_kc_to_cluster__(self.cluster_to_kcs)
        self.desc_to_cluster = self.__get_kc_to_cluster__(self.cluster_to_descs)
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.max_length_kc = args.max_length_kc
    def __get_kc_to_cluster__(self, cluster_to_kcs):

        kc_to_cluster = {}
        for cluster_id, kcs in cluster_to_kcs.items():
            for kc in kcs:
                kc_to_cluster[kc] = cluster_id
        return kc_to_cluster

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        problem = self.data[str(idx)]
        question = "[Q] " + problem['content']
        descs = ["[S] " + step for step in problem['descs']]
        concepts = ["[KC] " + kc for kc in problem['kcs']]

        question_enc = self.tokenizer(question, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        descs_encs = [self.tokenizer(step, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt') for step in descs]
        concept_encs = [self.tokenizer(kc, max_length=self.max_length_kc, truncation=True, padding='max_length', return_tensors='pt') for kc in concepts]

        kc_cluster_ids = [int(self.kc_to_cluster[kc]) for kc in problem['kcs']]
        kc_cluster_ids_tensors = torch.tensor(kc_cluster_ids, dtype=torch.long)
        desc_cluster_ids = [int(self.desc_to_cluster[desc]) for desc in problem['descs']]
        desc_cluster_ids_tensors = torch.tensor(desc_cluster_ids, dtype=torch.long)

        return {
            'question_ids': question_enc['input_ids'].squeeze(0),
            'question_mask': question_enc['attention_mask'].squeeze(0),
            'desc_ids': torch.cat([enc['input_ids'] for enc in descs_encs], dim=0),
            'desc_mask': torch.cat([enc['attention_mask'] for enc in descs_encs], dim=0),
            'kc_ids': torch.cat([enc['input_ids'] for enc in concept_encs], dim=0),
            'kc_mask': torch.cat([enc['attention_mask'] for enc in concept_encs], dim=0),
            'kc_cluster_ids': kc_cluster_ids_tensors,
            'desc_cluster_ids': desc_cluster_ids_tensors
        }

