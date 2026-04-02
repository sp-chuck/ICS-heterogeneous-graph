# Preprocess Report

## Config
- data_dir: /data/spz/data/swat_whole
- out_dir: /data/spz/AHGA
- train_file: train.csv
- list_file: list.txt
- graph_file: graph100.txt
- physical_threshold: 15.0
- corr_threshold: 0.75
- window_size: 4300
- balance_factor: 4.0
- directed: False
- with_self_loop: False
- export_homo_view: False
- temporal_mode: True
- temporal_window: 256
- temporal_stride: 128
- max_snapshots: 0
- seed: 42
- plc_rule: Infer by first digit of numeric code in device name: 1xx->PLC1 ... 6xx->PLC6
- crp_rule: Single virtual CRP0 connected to all virtual PLC1..PLC6

## Node Counts
- device: 51
- plc: 6
- crp: 1

## Edge Counts
- hierarchy: 114
- physical: 742
- flow: 114
- all: 970

## Quality
- isolated_nodes: 0
- isolated_ratio: 0.000000

## Column Alignment
- missing_in_train_vs_list: 0
- extra_in_train_vs_list: 0

## Model
- recommended_model: HTGNN
- architecture: Heterogeneous GNN encoder + GRU over snapshot sequence

## Temporal Summary
- num_snapshots: 370
- snapshot_label_normal: 370
- snapshot_label_attack: 0
- temporal_window: 256
- temporal_stride: 128