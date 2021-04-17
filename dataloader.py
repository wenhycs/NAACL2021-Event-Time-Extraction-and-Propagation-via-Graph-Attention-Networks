import json
import numpy as np
from datetime import datetime
from datetime import timedelta
import re
import calendar
from utils import TRAIN_DOCS, DEV_DOCS, TEST_DOCS, EDGE_LABELS
import torch
import dgl
from bisect import bisect


ENTITY_FILE = 'doc_ace_entities.json'
ACE_FILE = 'doc_ace.json'
TIME_FILE = 'data.json'
TEMP_FILE = "extracted_temporal_relation.json"


class Object(object):
    def to_dict(self):
        obj = self.__dict__
        for k, v in obj.items():
            if isinstance(v, Object):
                obj[k] = v.to_dict()
            elif isinstance(v, list) and len(v) and isinstance(v[0], Object):
                obj[k] = [_v.to_dict() for _v in v]
        return obj

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)


class EntityMention(Object):
    def __init__(self, end, start, text, type, id=None, sent_id=None, entity_id=None, role=None, **kwargs):
        self.end = end
        self.start = start
        self.text = text
        self.type = type
        self.id = id or entity_id
        if sent_id:
            self.sent_id = int(sent_id.split('-')[-1])
        else:
            self.sent_id = sent_id
        self.role = role
        self.embedding = None


class EventMention(Object):
    def __init__(self, triggers, arguments, id, event_type, label=None, **kwargs):
        assert len(triggers) == 1
        triggers = triggers[0]
        self.start = triggers['start']
        self.end = triggers['end']
        self.text = triggers['text']
        self.id = triggers['mention_id']
        self.event_id = id
        self.event_type = event_type
        self.arguments = []
        for argument in arguments:
            argument = EntityMention(**argument)
            self.arguments.append(argument)
        self.label = label
        self.embedding = None


def parse_timex(time_val):
    tmp = re.findall(r'[\d]{4}-[\d]{2}-[\d]{2}', time_val)
    if tmp:
        date = datetime.strptime(tmp[0], '%Y-%m-%d')
        return TwoTuple(date, date)
    tmp = re.findall(r'[\d]{4}-W[\d]{1,2}-[\d]{1}', time_val)
    if tmp:
        if tmp[0][-1] == '7':
            tmp[0] = tmp[0][:-1] + '0'
        date = datetime.strptime(tmp[0], '%Y-W%W-%w')
        return TwoTuple(date, date)
    tmp = re.findall(r'[\d]{4}-W[\d]{1,2}-WE', time_val)
    if tmp:
        early = datetime.strptime(tmp[0][:-3] + '-6', '%Y-W%W-%w')
        late = datetime.strptime(tmp[0][:-3] + '-0', '%Y-W%W-%w')
        return TwoTuple(early, late)
    tmp = re.findall(r'[\d]{4}-[\d]{2}', time_val)
    if tmp:
        date = datetime.strptime(tmp[0], '%Y-%m')
        _, num_days = calendar.monthrange(date.year, date.month)
        early = datetime.strptime(tmp[0] + '-1', '%Y-%m-%d')
        late = datetime.strptime(tmp[0] + f'-{num_days}', '%Y-%m-%d')
        return TwoTuple(early, late)
    tmp = re.findall(r'[\d]{4}-W[\d]{1,2}', time_val)
    if tmp:
        early = datetime.strptime(tmp[0] + '-1', '%Y-W%W-%w')
        late = datetime.strptime(tmp[0] + '-0', '%Y-W%W-%w')
        return TwoTuple(early, late)
    tmp = re.findall(r'[\d]{4}', time_val)
    if tmp:
        early = datetime.strptime(tmp[0] + '-01-01', '%Y-%m-%d')
        late = datetime.strptime(tmp[0] + '-12-31', '%Y-%m-%d')
        return TwoTuple(early, late)
    return None


class TimeArgument(EntityMention):
    def __init__(self, end, start, text, type, id, sent_id, timex, role=None, **kwargs):
        super().__init__(end=end, start=start, text=text, type=type, id=id, sent_id=sent_id, role=role)
        self.val = timex.get('val', None)
        self.anchor_dir = timex.get('anchor_dir', None)
        self.anchor_val = timex.get('anchor_val', None)
        if self.anchor_dir:
            if self.anchor_dir == 'AS_OF':
                self.val = self.anchor_val
                self.two_tuple = parse_timex(self.val)
            elif self.anchor_dir == 'BEFORE' and self.val == 'PAST_REF':
                self.val = self.anchor_val
                self.two_tuple = parse_timex(self.val)
                self.two_tuple.early = None
                self.two_tuple = None
            elif self.anchor_dir == 'AFTER' and self.val == 'FUTURE_REF':
                self.val = self.anchor_val
                self.two_tuple = parse_timex(self.val)
                self.two_tuple.late = None
                self.two_tuple = None
            else:
                self.two_tuple = None
        else:
            if self.val:
                self.two_tuple = parse_timex(self.val)
            else:
                self.two_tuple = None


def compare_time(t1, t2):
    if not t1:
        return True
    if not t2:
        return False
    return t1 <= t2


class TwoTuple(Object):
    def __init__(self, early=None, late=None):
        self.early = early
        self.late = late

    def all_inf(self):
        return not self.early and not self.late

    def equal(self, item):
        return self.early == self.late and self.early and \
               self.early == item.early and self.late == item.late

    def early_than(self, item):
        return compare_time(self.late, item.early)

    def to_dict(self):
        def to_str(x):
            if not x:
                return 'inf'
            else:
                return x.strftime('%Y-%m-%d')
        return {'early': to_str(self.early), 'late': to_str(self.late)}


class FourTuple(Object):
    def __init__(self, start_time=None, end_time=None, early_start=None, late_start=None, early_end=None, late_end=None):
        if start_time and end_time:
            self.start_time = start_time
            self.end_time = end_time
        else:
            self.start_time = TwoTuple(early_start, late_start)
            self.end_time = TwoTuple(early_end, late_end)

    def all_inf(self):
        return self.start_time.all_inf() and self.end_time.all_inf()


def convert_label_to_date(label):
    if label[0] == 0:
        return None
    else:
        return datetime.strptime(label[1], '%Y-%m-%d')


def parse_label(label):
    early_start = label['earliest_start_date']
    late_start = label['latest_start_date']
    early_end = label['earliest_end_date']
    late_end = label['latest_end_date']
    early_start = convert_label_to_date(early_start)
    late_start = convert_label_to_date(late_start)
    early_end = convert_label_to_date(early_end)
    late_end = convert_label_to_date(late_end)
    return FourTuple(early_start=early_start, late_start=late_start, early_end=early_end, late_end=late_end)


class Doc(Object):
    def __init__(self, doc_id, sent_offset, events, entity_mentions, labels, temporal_relation_map):
        self.id = doc_id
        self.sent_offset = sent_offset

        self.events = []
        all_argument_ids = set()
        for event_mention in events:
            event_mention = EventMention(**event_mention)
            self.events.append(event_mention)
            for argument in event_mention.arguments:
                all_argument_ids.add(argument.id)

        self.entity_mentions = []
        self.entity_pos = dict()
        self.time_arguments = []
        self.time_pos = dict()
        for mention_id, mention in entity_mentions.items():
            if mention['type'] == 'TIM:TIM':
                time_argument = TimeArgument(**mention)
                if time_argument.two_tuple:
                    self.time_pos[time_argument.id] = len(self.time_pos)
                    self.time_arguments.append(time_argument)
                else:
                    self.entity_pos[time_argument.id] = len(self.entity_mentions)
                    self.entity_mentions.append(time_argument)
            else:
                if mention_id not in all_argument_ids:
                    continue
                entity_mention = EntityMention(**mention)
                self.entity_pos[entity_mention.id] = len(self.entity_mentions)
                self.entity_mentions.append(entity_mention)

        for event in self.events:
            label = labels[event.event_id]
            event.label = parse_label(label)
        self.argument_graph = self.construct_argument_graph()
        self.extracted_temporal_graph = self.construct_extracted_temporal_graph(temporal_relation_map)
    
    def _dgl_graph_initialize(self):
        g = dgl.DGLGraph()
        g.add_nodes(len(self.events) + len(self.time_arguments) + len(self.entity_mentions))
        return g
    
    def _dgl_add_edges(self, g, edges):
        for i in range(g.number_of_nodes()):
            g.add_edge(i, i)
        for x, y, r in edges:
            g.add_edge(x, y)
        g.edata['w'] = torch.zeros(g.number_of_edges()).long()
        for i in range(g.number_of_nodes()):
            g.edata['w'][g.edge_id(i, i)] = EDGE_LABELS['SELF_LOOP']
        for x, y, r in edges:
            g.edata['w'][g.edge_id(x, y)] = r
        return g

    def get_argument_edges(self):
        edges = []
        visited = set()
        for i, event_mention in enumerate(self.events):
            for argument in event_mention.arguments:
                assert argument.id in self.time_pos or argument.id in self.entity_pos
                if argument.id in self.time_pos:
                    if not((i, len(self.events)+self.time_pos[argument.id]) in visited):
                        edges.append([i, len(self.events)+self.time_pos[argument.id], EDGE_LABELS[argument.role]])
                        edges.append([len(self.events)+self.time_pos[argument.id], i, EDGE_LABELS[argument.role]])
                        visited.add((i, len(self.events)+self.time_pos[argument.id]))
                else:
                    if not((i, len(self.events)+len(self.time_arguments)+self.entity_pos[argument.id]) in visited):
                        edges.append([i, len(self.events)+len(self.time_arguments)+self.entity_pos[argument.id], EDGE_LABELS[argument.role]])
                        edges.append([len(self.events)+len(self.time_arguments)+self.entity_pos[argument.id], i, EDGE_LABELS[argument.role]])
                        visited.add((i, len(self.events)+len(self.time_arguments)+self.entity_pos[argument.id]))
        return edges

    def get_extracted_temporal_edges(self, temporal_relation_map, with_time_arg=False):
        edges = []
        event_narrative_order = list(range(len(self.events)))
        event_narrative_order.sort(key=lambda x: self.events[x].start)
        for i in range(len(event_narrative_order) - 1):
            for j in range(i + 1, len(event_narrative_order)):
                x, y = event_narrative_order[i], event_narrative_order[j]
                sent_i, sent_j = bisect(self.sent_offset, self.events[x].start), bisect(self.sent_offset, self.events[y].start)
                if sent_j - sent_i > 1:
                    break
                if not((self.events[x].id, self.events[y].id) in temporal_relation_map):
                    continue
                else:
                    if temporal_relation_map[(self.events[x].id, self.events[y].id)] == 0:
                        edge_label, edge_label_inv = "BEFORE", "AFTER"
                    else: edge_label, edge_label_inv = "AFTER", "BEFORE"
                    edges.append([x, y, EDGE_LABELS[edge_label]])
                    edges.append([y, x, EDGE_LABELS[edge_label_inv]])
        if with_time_arg:
            visited = set()
            for i, event_mention in enumerate(self.events):
                for argument in event_mention.arguments:
                    assert argument.id in self.time_pos or argument.id in self.entity_pos
                    if argument.id in self.time_pos:
                        if not((i, len(self.events)+self.time_pos[argument.id]) in visited):
                            edges.append([i, len(self.events)+self.time_pos[argument.id], EDGE_LABELS[argument.role]])
                            edges.append([len(self.events)+self.time_pos[argument.id], i, EDGE_LABELS[argument.role]])
                            visited.add((i, len(self.events)+self.time_pos[argument.id]))
        return edges

    def construct_argument_graph(self):
        g = self._dgl_graph_initialize()
        edges = self.get_argument_edges()
        g = self._dgl_add_edges(g, edges)
        return g
    
    def construct_extracted_temporal_graph(self, temporal_relation_map):
        g = self._dgl_graph_initialize()
        edges = self.get_extracted_temporal_edges(temporal_relation_map, with_time_arg=True)
        g = self._dgl_add_edges(g, edges)
        return g


class InputFeature(Object):
    def __init__(self, embeddings, event_id, event_pos, time_id, time_pos, time_label, time_two_tuple, four_tuple, argument_graph, extracted_temporal_graph, all_times, all_time_pos):
        self.embeddings = embeddings
        self.event_id = event_id
        self.event_pos = event_pos
        self.time_id = time_id
        self.time_pos = time_pos
        self.time_label = time_label
        self.time_two_tuple = time_two_tuple
        self.four_tuple = four_tuple
        self.argument_graph = argument_graph
        self.extracted_temporal_graph = extracted_temporal_graph
        self.all_times = all_times
        self.all_time_pos = all_time_pos

    def to_tensor(self):
        self.embeddings = torch.FloatTensor(self.embeddings)
        self.event_pos = torch.LongTensor([self.event_pos])
        self.time_pos = torch.LongTensor([self.time_pos])
        self.time_label = torch.LongTensor(self.time_label)
        self.all_time_pos = torch.LongTensor(self.all_time_pos)

def creating_batch(data):
    embeddings, event_pos, time_pos, time_label = [], [], [], []
    argument_graph, extracted_temporal_graph = [], []
    total_len = 0
    for x in data:
        embeddings.append(x.embeddings)
        event_pos.append(x.event_pos + total_len)
        time_pos.append(x.time_pos + total_len)
        time_label.append(x.time_label)
        total_len += len(x.embeddings)
        argument_graph.append(x.argument_graph)
        extracted_temporal_graph.append(x.extracted_temporal_graph)
    embeddings = torch.cat(embeddings)
    event_pos = torch.cat(event_pos)
    time_pos = torch.cat(time_pos)
    time_label = torch.stack(time_label)
    argument_graph = dgl.batch(argument_graph)
    extracted_temporal_graph = dgl.batch(extracted_temporal_graph)
    return InputFeature(embeddings=embeddings,
                        event_id=None,
                        event_pos=event_pos,
                        time_id=None,
                        time_pos=time_pos,
                        time_label=time_label,
                        time_two_tuple=None, four_tuple=None,
                        argument_graph=argument_graph,
                        extracted_temporal_graph=extracted_temporal_graph,
                        all_times=None,
                        all_time_pos=None)

class EventDataset(Object):
    def __init__(self, doc_list):
        doc_entities = json.load(open(ENTITY_FILE))
        doc_ace = json.load(open(ACE_FILE))
        time_labels = json.load(open(TIME_FILE))
        extracted_temporal_relation = json.load(open(TEMP_FILE))
        temporal_relation_map = dict()
        for line_data in extracted_temporal_relation:
            if line_data["prob"] > 0.9:
                temporal_relation_map[(line_data["event_1"], line_data["event_2"])] = line_data["label"]

        self.docs = []
        for doc_id in doc_list:
            sent_offset = doc_ace[doc_id]['sent_offset']
            events = doc_ace[doc_id]['events']
            entity_mentions = doc_entities[doc_id]['entity_mentions']
            doc = Doc(doc_id, sent_offset, events, entity_mentions, time_labels, temporal_relation_map)
            self.docs.append(doc)

        self.feats = []
        for doc in self.docs:
            bert_cache = np.load(f'./longformer_cache/{doc.id}.npy')
            sent_offset = doc.sent_offset

            augmented_time_pos = [x + len(doc.events) for x in range(len(doc.time_pos))]

            embeddings = []
            for event in doc.events:
                event.embedding = bert_cache[event.start: event.end].mean(0)
                embeddings.append(event.embedding)

            for time in doc.time_arguments:
                time.embedding = bert_cache[sent_offset[time.sent_id] + time.start:
                                            sent_offset[time.sent_id] + time.end].mean(0)
                embeddings.append(time.embedding)

            for entity in doc.entity_mentions:
                entity.embedding = bert_cache[sent_offset[entity.sent_id] + entity.start:
                                              sent_offset[entity.sent_id] + entity.end].mean(0)
                embeddings.append(entity.embedding)

            for i, event in enumerate(doc.events):
                four_tuple = event.label
                for j, time in enumerate(doc.time_arguments):
                    two_tuple = time.two_tuple
                    labels = [int(four_tuple.start_time.early == two_tuple.early),
                              int(four_tuple.start_time.late == two_tuple.late),
                              int(four_tuple.end_time.early == two_tuple.early),
                              int(four_tuple.end_time.late == two_tuple.late)]

                    self.feats.append(InputFeature(embeddings, event.id, i, time.id, len(doc.events) + j, labels, two_tuple, four_tuple,\
                                                doc.argument_graph, doc.extracted_temporal_graph, doc.time_arguments, augmented_time_pos))

        self.indices = np.arange(len(self.feats))

    def get_dataloader(self):
        for feat in self:
            feat.to_tensor()
        return self

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.feats[item]

    def shuffle(self):
        np.random.shuffle(self.indices)

    def __iter__(self):
        for ix in self.indices:
            yield self[ix]

    def __add__(self, other):
        self.feats.extend(other.feats)
        self.indices = np.arange(len(self.feats))
        return self