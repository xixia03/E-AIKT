import torch
import json

#device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

kc_desc_map_path = ""
class Evaluator:
    def __init__(self, tokenizer, device, args):
        # Load both dataset and cluster files
        with open(args.json_file_dataset, 'r') as f:
            self.data_questions = json.load(f)
        with open(args.json_file_kc_questions, 'r') as file:
            self.kc_questions_map = json.load(file)
        with open(args.json_file_cluster_kc, 'r') as f:
            self.cluster_to_kcs = json.load(f)
        with open(kc_desc_map_path, 'r') as f:
            self.kc_desc_map = json.load(f)

        # Get kc to cluster mapping from cluster to kc
        self.kc_to_cluster = self.__get_kc_to_cluster__(self.cluster_to_kcs)
        self.kc_to_question = self.__get_kc_to_question__(self.data_questions)
        self.tokenizer = tokenizer
        self.eval_batch_size = args.eval_batch_size
        self.max_length = args.max_length
        self.max_length_kc = args.max_length_kc
        self.device = device

        self.__init_eval_dataset__()

    def __get_kc_to_cluster__(self, cluster_to_kcs):
       kc_to_cluster = {}
       for cluster_id, kcs in cluster_to_kcs.items():
           for kc in kcs:
               kc_to_cluster[kc] = cluster_id
       return kc_to_cluster
    def __get_kc_to_question__(self,data_questions):
        kc_to_question = []
        concept_index = 0  
        for question_idx, question_data in data_questions.items():
            question_index = int(question_idx)  
            for concept in question_data["kcs"]:
                kc_to_question.append([concept_index, question_index])
                concept_index += 1
        return torch.tensor(kc_to_question)
    
    def __init_eval_dataset__(self):
        self.list_kcs = [k for k in self.kc_questions_map]
        self.list_questions = [value['content'] for key, value in self.data_questions.items()]
        self.list_descs = [[sol for sol in value['descs']] for key,value in self.data_questions.items()]

        self.kcs = ['[KC]' + k for k in self.list_kcs]
        self.questions = ['[Q] ' + q for q in self.list_questions]
        self.descs = [['[S] ' + step for step in sol_steps] for sol_steps in self.list_descs]

    def sim_matrix(self, a, b, eps=1e-8):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.clamp(a_n, min=eps)
        b_norm = b / torch.clamp(b_n, min=eps)
        a_norm[torch.isnan(a_norm)] = 0
        b_norm[torch.isnan(b_norm)] = 0
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt
    def batched_sim_matrix(self, a, b, batch_size=512, eps=1e-8):
        sim_matrix = []
        for i in range(0, a.size(0), batch_size):
            a_batch = a[i:i + batch_size]
            sim_row = []
            for j in range(0, b.size(0), batch_size):
                b_batch = b[j:j + batch_size]
                sim_submatrix = self.sim_matrix(a_batch, b_batch, eps)
                sim_row.append(sim_submatrix)
            sim_matrix.append(torch.cat(sim_row, dim=1))
        return torch.cat(sim_matrix, dim=0)

    def __text_to_embeddings__(self, texts, model, max_length):
        embeddings = []
        model = model.module if hasattr(model, 'module') else model
        for i in range(0, len(texts), self.eval_batch_size):
            batch_texts = texts[i:i + self.eval_batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :])
        return torch.cat(embeddings, dim=0)

    def evaluate(self, model):
        model = model.module if hasattr(model, 'module') else model
        model.eval()

        question_embeddings = self.__text_to_embeddings__(self.questions, model, max_length=self.max_length)
        kc_embeddings = self.__text_to_embeddings__(self.kcs, model, max_length=self.max_length_kc)
        flat_descs = [step for sublist in self.descs for step in sublist]
        flat_desc_embeddings = self.__text_to_embeddings__(flat_descs, model, max_length=self.max_length)
        sol_desc_embeddings = []
        start_idx = 0
        for steps in self.descs:
            end_idx = start_idx + len(steps)
            sol_desc_embeddings.append(flat_desc_embeddings[start_idx:end_idx])
            start_idx = end_idx

        similarity_matrix_kc_question = self.batched_sim_matrix(kc_embeddings, question_embeddings)
        similarity_matrix_desc_question = self.batched_sim_matrix(flat_desc_embeddings,question_embeddings)
        similarity_matrix_desc_kc_question = []
        for kc,description in self.kc_desc_map.items():
            description = ['[S] ' + s for s in description]
            desc_indices = [flat_descs.index(desc) for desc in description if desc in flat_descs]
            if len(desc_indices) > 0:
                desc_similarities = similarity_matrix_desc_question[desc_indices, :]
                aggregated_similarity = desc_similarities.mean(dim=0)
            else:
                aggregated_similarity = torch.zeros(len(self.questions),device=self.device)
            similarity_matrix_desc_kc_question.append(aggregated_similarity)
        similarity_matrix_desc_kc_question = torch.stack(similarity_matrix_desc_kc_question)

        final_mean_similarity_matrix = (similarity_matrix_kc_question + similarity_matrix_desc_kc_question) / 2
        avg_acc, total_acc, avg_f1, micro_f1 = self.calculate_accuracies_clustered(final_mean_similarity_matrix)

        return avg_acc, total_acc, avg_f1, micro_f1 

    def calculate_accuracies_clustered(self, similarity_matrix):
        accuracies = {}
        total_correct = 0
        total_problems = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for kc_idx, kc in enumerate(self.list_kcs):
            cluster_kcs = self.cluster_to_kcs[self.kc_to_cluster[kc]]
            true_problem_indices = set()
            for cluster_kc in cluster_kcs:
                true_problem_indices.update(self.kc_questions_map[cluster_kc])
            num_problems = len(self.kc_questions_map[kc])
            total_problems += num_problems
            top_n_indices = torch.topk(similarity_matrix[kc_idx], k=num_problems).indices

            predicted_correct = sum([1 for idx in top_n_indices if idx.item() in true_problem_indices])
            total_correct += predicted_correct

            accuracy = predicted_correct / num_problems if num_problems > 0 else 0

            accuracies[kc] = {
                'total_problems': num_problems,
                'correct_predictions': predicted_correct,
                'accuracy': accuracy
            }

            top_n_indices = torch.topk(similarity_matrix[kc_idx], k=num_problems).indices

            tp = sum([1 for idx in top_n_indices if idx.item() in true_problem_indices])
            fp = num_problems - tp
            fn = num_problems - tp

            precision = tp / num_problems if num_problems > 0 else 0
            recall = tp / num_problems if num_problems > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            accuracies[kc].update({
                'f1': f1,
                'n': num_problems
            })

            total_tp += tp
            total_fp += fp
            total_fn += fn

        total_accuracy = total_correct / total_problems if total_problems > 0 else 0
        average_accuracy = sum([data['accuracy'] for data in accuracies.values()]) / len(accuracies) if accuracies else 0
        average_f1 = sum(data['f1'] for data in accuracies.values()) / len(accuracies) if accuracies else 0
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

        return average_accuracy, total_accuracy, average_f1, micro_f1
