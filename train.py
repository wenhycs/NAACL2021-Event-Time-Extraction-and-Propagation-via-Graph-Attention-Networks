import json
import copy
import sys
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch import nn
from RGAT import RGAT
from GAT import GAT
import torch
import random
import numpy as np
cuda_device_str = "cuda:0"
from tqdm import tqdm
import argparse
from utils import TRAIN_DOCS, DEV_DOCS, TEST_DOCS
import logging
import dgl
from optimization import AdamW
from dataloader import EventDataset, creating_batch
from datetime import datetime
import datetime as DT
import os


class Baseline(torch.nn.Module):
    def __init__(self, embed_size=768, bert_embed_size=768, hidden_size=384, dropout=0.):
        super(Baseline, self).__init__()
        self.embed_size = embed_size
        self.bert_embed_size = bert_embed_size
        self.hidden_size = hidden_size
        self.linear_1 = nn.Linear(self.bert_embed_size * 2, self.hidden_size)
        self.linear_2 = nn.Linear(self.hidden_size, 4)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, narrative_embedding, event_pos, time_pos):
        updated_embedding = self.dropout(narrative_embedding)
        event_repr = updated_embedding[event_pos]
        time_repr = updated_embedding[time_pos]
        event_time_pair = torch.cat([event_repr, time_repr], dim=1)
        logits = self.linear_2(self.dropout(torch.relu(self.linear_1(event_time_pair))))

        return logits



class EventGAT(torch.nn.Module):
    def __init__(self, embed_size=768, gcn_hidden=384, linear_hidden=384, dropout=0., gcn_layers=2, linear_layers=2, heads=4):
        super(EventGAT, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = gcn_hidden
        fan_in = embed_size
        if "rgat" in args.model:
            self.gat = RGAT(embed_size, gcn_hidden, gcn_hidden, heads, dropout, num_layers=gcn_layers)
            self.edge_embedding = nn.Embedding(41, 50)
        else:
            self.gat = GAT(embed_size, gcn_hidden, gcn_hidden, heads, dropout, num_layers=gcn_layers)

        self.linear_layers = nn.ModuleList()
        fan_in = gcn_hidden * 2
        for i in range(linear_layers - 1):
            linear_layer = nn.Linear(fan_in, linear_hidden)
            fan_in = gcn_hidden
            self.linear_layers.append(linear_layer)
        self.output_layer = nn.Linear(linear_hidden, 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, narrative_embedding, event_pos, time_pos, argument_graph=None):
        if "rgat" in args.model:
            argument_graph.edata['y'] = self.dropout(self.edge_embedding(argument_graph.edata['w']))
        hidden = self.dropout(narrative_embedding)
        hidden = self.gat(argument_graph, hidden)
        event_repr = hidden[event_pos]
        time_repr = hidden[time_pos]
        hidden = torch.cat([event_repr, time_repr], dim=1)
        for linear in self.linear_layers:
            hidden = self.dropout(torch.relu(linear(hidden)))
        logits = self.output_layer(hidden)

        return logits


def get_model(args):
    if args.model == 'baseline':
        return Baseline(hidden_size=args.hidden_size, dropout=args.dropout)
    elif args.model == 'event_gat':
        return EventGAT(gcn_hidden=args.gcn_hidden, dropout=args.dropout, gcn_layers=args.gcn_layers,
                        linear_layers=args.linear_layers, linear_hidden=args.hidden_size, heads=args.heads)
    elif args.model == 'event_rgat':
        return EventGAT(gcn_hidden=args.gcn_hidden, dropout=args.dropout, gcn_layers=args.gcn_layers,
                        linear_layers=args.linear_layers, linear_hidden=args.hidden_size, heads=args.heads)
    elif args.model == 'temp_gat':
        return EventGAT(gcn_hidden=args.gcn_hidden, dropout=args.dropout, gcn_layers=args.gcn_layers,
                        linear_layers=args.linear_layers, linear_hidden=args.hidden_size, heads=args.heads)
    elif args.model == 'temp_rgat':
        return EventGAT(gcn_hidden=args.gcn_hidden, dropout=args.dropout, gcn_layers=args.gcn_layers,
                        linear_layers=args.linear_layers, linear_hidden=args.hidden_size, heads=args.heads)


def get_input_data(data):
    if 'baseline' == args.model:
        narrative_embedding = data.embeddings.to(device)
        event_pos = data.event_pos.to(device)
        time_pos = data.time_pos.to(device)
        input_data = {
            'narrative_embedding': narrative_embedding,
            'event_pos': event_pos,
            'time_pos': time_pos,
        }
    elif args.model == 'event_gat' or args.model == 'event_rgat':
        narrative_embedding = data.embeddings.to(device)
        argument_graph = data.argument_graph.to(device)
        event_pos = data.event_pos.to(device)
        time_pos = data.time_pos.to(device)
        input_data = {
            'narrative_embedding': narrative_embedding,
            'event_pos': event_pos,
            'time_pos': time_pos,
            'argument_graph': argument_graph,
        }
    elif args.model == 'temp_rgat' or args.model == 'temp_gat':
        narrative_embedding = data.embeddings.to(device)
        argument_graph = data.extracted_temporal_graph.to(device)
        event_pos = data.event_pos.to(device)
        time_pos = data.time_pos.to(device)
        input_data = {
            'narrative_embedding': narrative_embedding,
            'event_pos': event_pos,
            'time_pos': time_pos,
            'argument_graph': argument_graph,
        }
    return input_data


def predict_time(event_id_scores, threshold):
    all_preds = []
    all_gt = []
    for _, event in event_id_scores.items():
        pred = []
        for i, s in enumerate(event['predicted_scores']):
            if s < threshold:
                predicted = None
            else:
                predicted = event['predicted_time'][i]
            pred.append(predicted)
        all_preds.append(pred)
        all_gt.append(event['ground_truth'])

    cnt = {
        'gt_inf': {
            'total': 0.,
            'hit': 0.,
            'score': 0.,
        },
        'gt_normal': {
            'total': 0,
            'hit': 0,
            'score': 0,
        },
        'pred_inf': 0.
    }

    for predicted, ground_truth in zip(all_preds, all_gt):
        for pred, gt in zip(predicted, ground_truth):
            if not gt:
                cnt['gt_inf']['total'] += 1
                if pred == gt:
                    cnt['gt_inf']['hit'] += 1
                    cnt['gt_inf']['score'] += 1
            else:
                cnt['gt_normal']['total'] += 1
                if pred == gt:
                    cnt['gt_normal']['hit'] += 1
                    cnt['gt_normal']['score'] += 1
                elif pred:
                    cnt['gt_normal']['score'] += 1. / (1. + abs((gt - pred).days))

            if not pred:
                cnt['pred_inf'] += 1

    cnt['gt_inf']['acc'] = cnt['gt_inf']['hit'] / cnt['gt_inf']['total']
    cnt['gt_inf']['f_acc'] = cnt['gt_inf']['score'] / cnt['gt_inf']['total']
    cnt['gt_normal']['acc'] = cnt['gt_normal']['hit'] / cnt['gt_normal']['total']
    cnt['gt_normal']['f_acc'] = cnt['gt_normal']['score'] / cnt['gt_normal']['total']
    cnt['acc'] = (cnt['gt_inf']['hit'] + cnt['gt_normal']['hit']) / \
                 (cnt['gt_inf']['total'] + cnt['gt_normal']['total'])
    cnt['f_acc'] = (cnt['gt_inf']['score'] + cnt['gt_normal']['score']) / \
                   (cnt['gt_inf']['total'] + cnt['gt_normal']['total'])
    return cnt, np.array([event['predicted_scores'] for event in event_id_scores.values()])


def save_event_id_scores(event_id_scores, th):
    for event_id in event_id_scores:
        for k in range(len(event_id_scores[event_id]['predicted_time'])):
            if event_id_scores[event_id]['ground_truth'][k]:
                event_id_scores[event_id]['ground_truth'][k] = event_id_scores[event_id]['ground_truth'][k].strftime("%Y-%m-%d")
            else: event_id_scores[event_id]['ground_truth'][k] = "inf"
            if event_id_scores[event_id]['predicted_time'][k] != None and event_id_scores[event_id]['predicted_scores'][k] > th:
                event_id_scores[event_id]['predicted_time'][k] = event_id_scores[event_id]['predicted_time'][k].strftime("%Y-%m-%d")
            else: event_id_scores[event_id]['predicted_time'][k] = "inf"
            if event_id_scores[event_id]['ground_truth'][k] == event_id_scores[event_id]['predicted_time'][k]:
                event_id_scores[event_id]['true_label'][k] = 1.0
            else: event_id_scores[event_id]['true_label'][k] = 0.0
    f = open("./outputs/"+args.run_name+"_4tuple_baseline_max_scores.json", "w")
    f.write(json.dumps(event_id_scores, indent=2))
    f.close()


def evaluate(model, loader, save_file=False, threshold=None, batch_size=1):
    model.eval()
    all_scores = []
    all_labels = []
    event_id_scores = {}

    total_instances = len(loader)
    total_steps = int(total_instances / batch_size) + int((total_instances % batch_size)==0)
    for start in tqdm(range(0, total_instances, batch_size), total=total_steps, desc='eval'):
        end = min(start+batch_size, total_instances-1)
        data = loader[start:end]
        data = creating_batch(data)
        input_data = get_input_data(data)
        time_labels = data.time_label.numpy()

        scores = model(**input_data)
        scores = scores.detach().cpu().numpy()
        for i in range(len(scores)):
            all_labels.append(time_labels[i])
            all_scores.append(scores[i])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    criterion = nn.BCEWithLogitsLoss()
    eval_loss = criterion(torch.tensor(all_scores.flatten()).double(),
                          torch.tensor(all_labels.flatten()).double()).item()
    for data, scores, labels in zip(loader, all_scores, all_labels):
        event_id = data.event_id
        time_id = data.time_id

        for i, s in enumerate(scores):
            if event_id not in event_id_scores:
                event_id_scores[event_id] = {"predicted_scores": [-1e9] * 4,
                                             "predicted_id": [None] * 4,
                                             "predicted_time": [None] * 4,
                                             "true_label": [0] * 4,
                                             "ground_truth": [data.four_tuple.start_time.early,
                                                              data.four_tuple.start_time.late,
                                                              data.four_tuple.end_time.early,
                                                              data.four_tuple.end_time.late]}

            if s > event_id_scores[event_id]["predicted_scores"][i]:
                event_id_scores[event_id]["predicted_scores"][i] = float(s)
                event_id_scores[event_id]["predicted_id"][i] = time_id
                event_id_scores[event_id]["true_label"][i] = float(labels[i])
                if i % 2 == 0:
                    predicted = data.time_two_tuple.early
                else:
                    predicted = data.time_two_tuple.late
                event_id_scores[event_id]['predicted_time'][i] = predicted

    cnt, max_scores = predict_time(event_id_scores, threshold=threshold or args.threshold)

    cnt['loss'] = eval_loss

    if threshold is None:
        import time
        tic = time.time()
        logging.info(json.dumps(cnt, indent=2))
        scores = max_scores.flatten()
        scores = np.unique(np.sort(scores))
        best_th = None
        best_acc = 0
        best_reuslt = None
        for th in scores:
            tmp_result, _ = predict_time(event_id_scores, th)
            if tmp_result['acc'] > best_acc:
                best_acc = tmp_result['acc']
                best_reuslt = tmp_result
                best_th = th
        toc = time.time()
        logging.info(f'***** threshold = {best_th}, time elapsed = {toc - tic} *****')
        cnt = best_reuslt
        cnt['loss'] = eval_loss
        logging.info(json.dumps(cnt, indent=2))
        cnt['max_scores'] = max_scores
        if save_file:
            save_event_id_scores(event_id_scores, best_th)
        return best_reuslt, best_th
    else:
        cnt['loss'] = eval_loss
        logging.info(json.dumps(cnt, indent=2))
        cnt['max_scores'] = max_scores
        return cnt, event_id_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='baseline', type=str, choices=['baseline', 'event_gat', 'event_rgat', 'temp_gat', 'temp_rgat'])
    parser.add_argument('--run_name', default=None, type=str)

    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--train_epochs', default=500, type=int)
    parser.add_argument('--weight_decay', default=0., type=float)

    parser.add_argument('--hidden_size', default=384, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)

    parser.add_argument('--gcn_layers', default=2, type=int)
    parser.add_argument('--gcn_hidden', default=384, type=int)
    parser.add_argument('--linear_layers', default=2, type=int)

    parser.add_argument("--heads", default=4, type=int)

    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--threshold", default=0., type=float)

    args = parser.parse_args()

    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    if not os.path.exists('./ckpt/'):
        os.makedirs('./ckpt/')

    if args.eval:
        train, dev, test = EventDataset(TRAIN_DOCS), EventDataset(DEV_DOCS), EventDataset(TEST_DOCS)
        device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")
        model = get_model(args).to(device)
        testloader = test.get_dataloader()
        devloader = dev.get_dataloader()
        model.load_state_dict(torch.load("./ckpt/"+args.run_name+".pt"))
        model.eval()
        dev_result, threshold = evaluate(model, devloader)
        test_result, event_id_scores = evaluate(model, testloader, save_file=True, threshold=threshold)
        print(f"test_acc = {test_result['acc']}, test_f_acc = {test_result['f_acc']}")
        save_event_id_scores(event_id_scores, threshold)
    else:
        logging.basicConfig(level=logging.INFO, filename=f'results/{args.run_name}.log', filemode='w', format='%(message)s')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        seed = args.seed
        batch_size = args.batch_size

        logging.info(str(json.dumps(args.__dict__, indent=2, ensure_ascii=False)))
        device = torch.device(cuda_device_str if torch.cuda.is_available() else "cpu")
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        dgl.random.seed(seed)
        train, dev, test = EventDataset(TRAIN_DOCS), EventDataset(DEV_DOCS), EventDataset(TEST_DOCS)
        model = get_model(args).to(device)
        for k, v in model.named_parameters():
            logging.info(k + str(list(v.size())))

        if args.weight_decay > 0:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        trainloader = train.get_dataloader()
        devloader = dev.get_dataloader()
        testloader = test.get_dataloader()

        max_acc, test_acc = 0., 0.
        train_loss = []
        for i in tqdm(range(1, args.train_epochs + 1), desc='epoch'):
            trainloader.shuffle()
            total_loss = 0.
            total_instances = len(trainloader)
            total_steps = int(total_instances / batch_size) + int((total_instances % batch_size)==0)
            for start in tqdm(range(0, total_instances, batch_size), total=total_steps, desc='batch'):
                model.train()
                model.zero_grad()
                end = min(start+batch_size, total_instances-1)
                data = trainloader[start:end]
                data = creating_batch(data)
                input_data = get_input_data(data)
                time_labels = data.time_label.to(device)

                scores = model(**input_data).squeeze()
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(scores.view(-1, 4), time_labels.view(-1, 4).float())
                total_loss += float(loss.detach().cpu())
                loss.backward()
                optimizer.step()

            dev_result, threshold = evaluate(model, devloader, batch_size=batch_size)
            logging.info(f'train_loss: {total_loss / total_steps}, dev_acc: {dev_result["acc"]}, '
                         f'dev_qacc: {dev_result["f_acc"]}, dev_loss: {dev_result["loss"]}, ')
            dev_acc = dev_result['acc']
            if dev_acc > max_acc:
                logging.info(f"New Best, dev_acc = {dev_acc}")
                max_acc = dev_acc
                torch.save(model.state_dict(),"./ckpt/"+args.run_name+".pt")
