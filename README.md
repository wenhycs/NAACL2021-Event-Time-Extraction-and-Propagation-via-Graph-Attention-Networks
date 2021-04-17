# Event Time Extraction and Propagation via Graph Attention Networks

Resources for paper "Event Time Extraction and Propagation via Graph Attention Networks". NAACL 2021

## Data
The 4-tuple annotations are in `data.json`. The data are at event level since the time representations for all coreferential mentions should be consistent.

Because of the license restriction, we are not able to provide pre-processed ACE2005 data (`doc_ace_entities.json`, `doc_ace.json`).

## Dependencies
- torch=1.2.0
- tqdm=4.36.1
- transformers=3.3.0
- numpy=1.18.4
- dgl-cuda10.0=0.4.3post2
