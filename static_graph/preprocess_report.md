# Preprocess Report

## Config
- data_dir: /data/spz/data/swat_whole
- out_dir: /data/spz/AHGA
- static_out_dir: /data/spz/AHGA/static_graph
- static_subdir: static_graph
- train_file: train.csv
- test_file: test.csv
- list_file: list.txt
- graph_file: graph100.txt
- physical_threshold: 15.0
- corr_threshold: 0.75
- window_size: 4300
- balance_factor: 4.0
- feature_version: v3
- legacy_window_mode: False
- legacy_block_size: 100
- legacy_device_indexes: 1,2,3,4,6,7,8,9,10,13,15,17,18,19,20,21,22,23,25,26,27,28,29,35,36,37,39,40,41,42,45,46,47,48,50
- period_w0: 16
- period_w_max: 256
- period_delta: 4
- period_delta_w: 2
- entropy_bins: 16
- constant_threshold: 1e-06
- large_period_value: 1000000.0
- period_convexity_drop_ratio: 0.02
- period_min_autocorr: 0.3
- reuse_node_features: False
- reuse_node_features_file: 
- v3_device_types: ['AIT', 'DPIT', 'FIT', 'LIT', 'MV', 'P', 'PIT', 'UNKNOWN', 'UV']
- v3_processes: ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'PUNK']
- v3_sub_processes: ['SP10', 'SP20', 'SP30', 'SP40', 'SP50', 'SP60', 'SPUNK']
- directed: False
- with_self_loop: False
- export_homo_view: False
- temporal_mode: True
- temporal_window: 256
- temporal_stride: 128
- max_snapshots: 0
- build_test_temporal: True
- test_temporal_subdir: temporal_snapshots_test
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