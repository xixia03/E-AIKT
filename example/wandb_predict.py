import os
import argparse
import json
import copy
import torch
import pandas as pd
import sys
sys.path.insert(0,'')
sys.path.append('..')
from pykt.models import evaluate,evaluate_question,load_model, evaluate_testset
from pykt.datasets import init_test_datasets

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

with open("../configs/wandb.json") as fin:
    wandb_config = json.load(fin)

def main(params):
    if params['use_wandb'] ==1:
        import wandb
        if "wandb_project_name" in params and params["wandb_project_name"] != "":
            wandb.init(project=params["wandb_project_name"])
        else:
            wandb.init()

    save_dir, batch_size, fusion_type = params["save_dir"], params["bz"], params["fusion_type"].split(",")

    ckpt_path = ""
    save_dir = os.path.join(save_dir, ckpt_path)
    win200 = params["win200"]

    with open(os.path.join(save_dir, "config.json")) as fin:
        config = json.load(fin)
        model_config = copy.deepcopy(config["model_config"])
        for remove_item in ['use_wandb','learning_rate','add_uuid','l2']:
            if remove_item in model_config:
                del model_config[remove_item]   
        if "emb_path" in model_config:
            del model_config["emb_path"]
        trained_params = config["params"]
        model_name, dataset_name, emb_type, fold = trained_params["model_name"], trained_params["dataset_name"], trained_params["emb_type"], trained_params["fold"]
        if model_name in ["saint", "sakt", "cdkt"]:
            train_config = config["train_config"]
            seq_len = train_config["seq_len"]
            model_config["seq_len"] = seq_len   

        if model_name in ["stosakt"]:
            train_args = argparse.ArgumentParser()
            print(f"train_args;{train_args}")


    with open("../configs/data_config.json") as fin:
        curconfig = copy.deepcopy(json.load(fin))
        if model_name in ["eaikt"]:
            dataset_name = params["dataset_name"]
        data_config = curconfig[dataset_name]
        data_config["dataset_name"] = dataset_name
        if model_name in ["dkt_forget", "bakt_time"] or emb_type.find("time") != -1:
            data_config["num_rgap"] = config["data_config"]["num_rgap"]
            data_config["num_sgap"] = config["data_config"]["num_sgap"]
            data_config["num_pcount"] = config["data_config"]["num_pcount"]
        elif model_name in ["lpkt"]:
            print("running  prediction")
            data_config["num_at"] = config["data_config"]["num_at"]
            data_config["num_it"] = config["data_config"]["num_it"] 
        elif model_name in ["eaikt"]:
            data_config["num_q"] = config["data_config"]["num_q"]
            data_config["num_c"] = config["data_config"]["num_c"] 
    
    # Add emb_path from params to data config, if model used emb_path from params
    if "emb_path" in trained_params:
        data_config["emb_path"] = trained_params["emb_path"]
    
    test_loader, test_window_loader, test_question_loader, test_question_window_loader = init_test_datasets(data_config, model_name, batch_size,fold,win200,params['pretrain_suffix'])
    print(f"Start predicting model: {model_name}, embtype: {emb_type}, save_dir: {save_dir}, dataset_name: {dataset_name}")
    print(f"model_config: {model_config}")
    print(f"data_config: {data_config}")

    if model_name in ["stosakt"]:
        model = load_model(model_name, model_config, data_config, emb_type, save_dir, train_args)
    else:

        for remove_item in ['use_wandb','learning_rate','add_uuid','l2','global_bs','num_gpus','pretrain_path', 'num_epochs', 'batch_size']:
            if remove_item in model_config:
                del model_config[remove_item]
        if params["load_finetune"] == "1":
            model = load_model(model_name, model_config, data_config, emb_type, save_dir, args=args, mode="train", finetune=True)
        else:
            print(f"loading model from {save_dir} to evaluate ...")
            model= load_model(model_name, model_config, data_config, emb_type, save_dir, mode="test")
    model.to(device)
    save_test_path = ""
    if params["save_all_preds"] == 1:
        save_test_path = os.path.join(save_dir, model.emb_type+"_test_predictions.txt")
        print(f"Predictions will be saved to {save_test_path}")

    if model.model_name == "rkt":
        dpath = data_config["dpath"]
        dataset_name = dpath.split("/")[-1]
        tmp_folds = set(data_config["folds"]) - {fold}
        folds_str = "_" + "_".join([str(_) for _ in tmp_folds])
        if dataset_name in ["algebra2005", "bridge2algebra2006"]:
            fname = "phi_dict" + folds_str + ".pkl"
            rel = pd.read_pickle(os.path.join(dpath, fname))
        else:
            fname = "phi_array" + folds_str + ".pkl" 
            rel = pd.read_pickle(os.path.join(dpath, fname))                

    if model.model_name == "rkt":
        testauc, testacc = evaluate(model, test_loader, model_name, rel, save_path=save_test_path)
    else:
        testauc, testacc = evaluate(model, test_loader, model_name, save_path=save_test_path)
    print(f"testauc: {testauc}, testacc: {testacc}")

    if params["save_all_preds"] == 1:
        save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
        print(f"Predictions will be saved to {save_test_window_path}")
    
    if dataset_name in ["statics2011", "assist2015", "poj"] or model_name in ["lpkt", "gnn4kt"]:
        save_test_window_path = os.path.join(save_dir, model.emb_type+"_test_window_predictions.txt")
        window_testauc, window_testacc = evaluate(model, test_window_loader, model_name, save_test_window_path, dataset_name, fold)
        print(f"window_testauc: {window_testauc}, window_testacc: {window_testacc}")

    question_testauc, question_testacc = -1, -1
    question_window_testauc, question_window_testacc = -1, -1
  
    dres = {
        "testauc": testauc, "testacc": testacc,
    }  

    if "test_question_file" in data_config and not test_question_loader is None:
        save_test_question_path = os.path.join(save_dir, model.emb_type+"_test_question_predictions.txt")
        q_testaucs, q_testaccs = evaluate_question(model, test_question_loader, model_name, fusion_type, save_path=save_test_question_path)
        for key in q_testaucs:
            dres["oriauc"+key] = q_testaucs[key]
        for key in q_testaccs:
            dres["oriacc"+key] = q_testaccs[key]
            
    if "test_question_window_file" in data_config and not test_question_window_loader is None:
        save_test_question_window_path = os.path.join(save_dir, model.emb_type+"_test_question_window_predictions.txt")
        qw_testaucs, qw_testaccs = evaluate_question(model, test_question_window_loader, model_name, fusion_type, save_path=save_test_question_window_path)
        for key in qw_testaucs:
            dres["windowauc"+key] = qw_testaucs[key]
        for key in qw_testaccs:
            dres["windowacc"+key] = qw_testaccs[key]

    print(f"testauc: {testauc}, testacc: {testacc}")
    print(f"question_testauc: {question_testauc}, question_testacc: {question_testacc}, question_window_testauc: {question_window_testauc}, question_window_testacc: {question_window_testacc}")
    
    print(dres)
    results_save_path = os.path.join(save_dir, "prediction_results.json")
    with open(results_save_path, 'w') as json_file:
        json.dump(dres, json_file, indent=2)

    raw_config = json.load(open(os.path.join(save_dir,"config.json")))
    dres.update(raw_config['params'])
    dres["dataset_name"] = dataset_name

    if params['use_wandb'] ==1:
        wandb.log(dres)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bz", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="saved_model_xes3g5m")
    parser.add_argument("--fusion_type", type=str, default="late_fusion")
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--wandb_project_name", type=str, default="default")
    parser.add_argument("--save_all_preds", type=int, default=1, help="1 for saving all predictions/ground truths as list: takes a lot of time and space.")
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--pretrain_suffix", type=str, default="pretrain")
    parser.add_argument("--win200", type=bool, default=False)
    parser.add_argument("--load_finetune", type=str, default="0")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--pretrain_model", type=int, default=0)
    parser.add_argument("--finetune_dataset_name", type=str, default='None')
    parser.add_argument("--apply_mask", type=str, default='None')
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fold", type=str, default='4')

    args = parser.parse_args()
    print(args)
    params = vars(args)
    
    main(params)
