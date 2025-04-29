import json
from collections import defaultdict

data_path = ""
kc_question_path = ""
kc_desc_path = ""

with open(data_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

concept_to_problems = {}
concept_to_descs = defaultdict(list)

for problem_id, problem_content in data.items():
    concepts = problem_content.get("kcs", [])
    descs = problem_content.get("descs", [])
    for concept in concepts:
        if concept not in concept_to_problems:
            concept_to_problems[concept] = []
        concept_to_problems[concept].append(int(problem_id))
    for kc, desc in zip(concepts, descs):
        concept_to_descs[kc].append(desc)

concept_to_descs = dict(concept_to_descs)
formatted_concept_to_problems = {
    concept: [problem_id for problem_id in problem_ids]
    for concept, problem_ids in concept_to_problems.items()
}

with open(kc_question_path, 'w', encoding='utf-8') as f:
    json.dump(formatted_concept_to_problems, f, ensure_ascii=False, indent=4)

with open(kc_desc_path, 'w', encoding='utf-8') as f:
    json.dump(concept_to_descs, f, ensure_ascii=False, indent=4)

