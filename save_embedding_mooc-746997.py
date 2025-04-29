from transformers import BertModel
from transformers import AutoTokenizer,AutoConfig
import torch
import json
from tqdm import tqdm
import os
from sklearn.preprocessing import normalize

folder_saved_model = ''
path_data_questions = ''
path_kc_questions_map = ''

with open(path_data_questions, 'r') as file:
    data_questions = json.load(file)
with open(path_kc_questions_map, 'r') as file:
    kc_questions_map = json.load(file)

embeddings_save_folder = ""

if not os.path.exists(embeddings_save_folder):
    os.makedirs(embeddings_save_folder)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(folder_saved_model + '/tokenizer')
# Create a configuration object or load it if you have saved one
config = AutoConfig.from_pretrained('')
# Initialize the model with this configuration
model = BertModel(config)
# Adjust the model's token embeddings to account for new tokens before loading the weights
model.resize_token_embeddings(len(tokenizer))
# Load the model weights
model.load_state_dict(torch.load(folder_saved_model + ''))
# Move the model to the appropriate computing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Set the model to training or evaluation mode as needed
model = model.eval()  # or model.train() if you continue training

BATCH_SIZE = 512  # Define your batch size here

# Helper function to batch text data and convert to embeddings
def text_to_embeddings(texts, max_length=128):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating Embeddings"):
        batch_texts = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :])  # Extract [CLS] token embeddings
    return torch.cat(embeddings, dim=0)

list_questions = [value['content'] for key, value in data_questions.items()]
list_descs = [[sol for sol in value['descs']] for key,value in data_questions.items()]
questions = ['[Q] ' + q for q in list_questions]
kcs = [['[S] ' + kc for kc in sol_steps] for sol_steps in list_descs]

question_embeddings = text_to_embeddings(questions)
flat_kcs = [step for sublist in kcs for step in sublist]
flat_kc_embeddings = text_to_embeddings(flat_kcs)

# Map flat embeddings back to their respective lists using original lengths
sol_step_embeddings = []
start_idx = 0
for steps in kcs:
    end_idx = start_idx + len(steps)
    sol_step_embeddings.append(flat_kc_embeddings[start_idx:end_idx])
    start_idx = end_idx

np_question_embeddings = question_embeddings.cpu().detach().numpy()
np_kc_embeddings = []
for i in range(len(sol_step_embeddings)):
    np_kc_embeddings.append(sol_step_embeddings[i].cpu().detach().numpy())

np_desc_embeddings_mean = []
for i in range(len(np_kc_embeddings)):
    np_desc_embeddings_mean.append(np_kc_embeddings[i].mean(axis=0))

dict_emb = {}
for idx, value in enumerate(data_questions.values()):
    emb_q = np_question_embeddings[idx].copy().reshape(1,-1)
    emb_sol = np_desc_embeddings_mean[idx].copy().reshape(1,-1)
    emb = (emb_q + emb_sol)/2

    norm_emb = normalize(emb, axis=1, norm='l2').flatten()
    dict_emb[value['qid']] = norm_emb.tolist()

save_path = os.path.join(embeddings_save_folder, '')

with open(save_path, 'w') as f:
    json.dump(dict_emb, f)