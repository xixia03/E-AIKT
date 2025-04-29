from zhipuai import ZhipuAI
import json
import time

original_question_file = ""
annotated_question_file = ""
ZHIPU_API_KEY = ""
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# dbe-kt22 dataset cot prompt
system_prompt = """you will be provided with a question related to a relational database course for undergraduate and graduate students, along with its final answer. Your task is to generate a concise and necessary list of knowledge concepts required to correctly solve the question using the following structured approach:

1. Problem Analysis: Analyze the problem, extract known conditions and the target to be solved.
2. Reasoning Path Decomposition: Based on the problem, answer, and explanation, decompose the reasoning path in the following format:
	- The calculation and reasoning task for each step.
	- The knowledge concepts necessary for each step.
	- The logic and relationship between each step and the preceding and following steps.
3. Knowledge Concept Extraction: Based on the above reasoning path, extract the knowledge concepts required to solve the problem and extract the following content.:
	- The knowledge concepts necessary for correctly solving the problem and their application descriptions, separated by a colon.
	- Do not generate any other text apart from the knowledge concept summary, and do not use bullet points. Separate each concept with a newline.
	- Only provide the essential knowledge concepts, avoiding redundancy, repetition, or unnecessary content. Prioritize necessity and extreme conciseness.
Only return the knowledge concepts and application descriptions extracted in step 3.
"""

user_prompt_template = """question: {}
Final Answer: {}
Explanation: {}"""

# def structure_answer(item):
#     if item["type"] == "填空题"  or item["type"] == "判断题":
#         return item["answer"]
#     elif item["type"] == "单选题":
#         choice = item["answer"][0]
#         return f"{choice}: {item['options'][choice]}"
#     elif item["type"] == "多选题":
#         options = item["options"]  # 获取选项字典
#         answers = item["answer"]  # 获取正确答案列表
#         result = ', '.join([f"{ans}: {options[ans]}" for ans in answers])
#         return result
def create_full_user_prompt(item):
    #dbe-kt22
    return user_prompt_template.format(item['content'], item["answer"])
    # xes3g5m
    # answer_structured = structure_answer(item)
    # return user_prompt_template.format(item['content'], answer_structured, item["analysis"])
def get_kcs(item):
    full_user_prompt = create_full_user_prompt(item)
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": full_user_prompt
            }
        ],
        temperature=0,
        max_tokens=256,
        top_p=1
    )
    return response.choices[0].message.content

with open(original_question_file, 'r', encoding='utf-8') as file:
    data = json.load(file)
counter = 0

for index, (key, item) in enumerate(data.items()):
    if index == 374:
        continue
    start_time = time.time()
    item['knowledge_concepts_text'] = get_kcs(item)
    lines = item['knowledge_concepts_text'].split('\n')
    kc_list = []
    kc_descriptions_list = []
    for line in lines:
        if ':' in line:
            concept, description = line.split(':', 1)
            kc_list.append(concept.strip())
            kc_descriptions_list.append(description.strip())

    item['kcs'] = kc_list
    item['descs'] = kc_descriptions_list
    end_time = time.time()  # Capture the end time
    iter_time = end_time - start_time  # Calculate the time taken for this iteration
    print(f"The question {key} took {iter_time:.2f} seconds to convert")
    counter += 1  # Increment the counter
    # Save the progress at every 100 iterations
    if counter % 4 == 0:
        with open(annotated_question_file, 'w', encoding='utf-8') as temp_file:
            json.dump(data, temp_file, ensure_ascii=False, indent=2)
        print(f"Progress saved at iteration {counter}")

with open(annotated_question_file, 'w', encoding='utf-8') as temp_file:
    json.dump(data, temp_file, ensure_ascii=False, indent=2)
print(f"Progress saved at iteration {counter}")