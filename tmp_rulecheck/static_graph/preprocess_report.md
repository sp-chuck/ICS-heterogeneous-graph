# Preprocess Report

## Config
- data_dir: /data/spz/data/swat_whole
- out_dir: /data/spz/AHGA/tmp_rulecheck
- static_out_dir: /data/spz/AHGA/tmp_rulecheck/static_graph
- static_subdir: static_graph
- train_file: train.csv
- test_file: test.csv
- list_file: list.txt
- graph_file: graph100.txt
- physical_threshold: 15.0
- corr_threshold: 0.75
- flow_rule_mode: rule_based_no_pearson
- period_similarity_threshold: 0.9
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
- reuse_node_features: True
- reuse_node_features_file: /data/spz/AHGA/static_graph/hetero_node_features.csv
- v3_device_types: ['AIT', 'DPIT', 'FIT', 'LIT', 'MV', 'P', 'PIT', 'UNKNOWN', 'UV']
- v3_processes: ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'PUNK']
- v3_sub_processes: ['SP10', 'SP20', 'SP30', 'SP40', 'SP50', 'SP60', 'SPUNK']
- directed: False
- with_self_loop: False
- export_homo_view: False
- temporal_mode: False
- temporal_window: 256
- temporal_stride: 128
- max_snapshots: 0
- build_test_temporal: False
- test_temporal_subdir: temporal_snapshots_test
- seed: 42
- plc_rule: Infer by first digit of numeric code in device name: 1xx->PLC1 ... 6xx->PLC6
- crp_rule: Single virtual CRP0 connected to all virtual PLC1..PLC6
- flow_rules: ['FITs from adjacent stages => true', 'FITs and LIT/LT within the same stage => true', 'Adjacent devices in piping diagram (name-based) => true', 'AITs and other sensors => false', 'Devices in same stage with similar periodic features => true']

## Node Counts
- device: 51
- plc: 6
- crp: 1

## Edge Counts
- hierarchy: 114
- physical: 742
- flow: 252
- all: 1108

## Flow Rule Stats
- adjacent_stage_fit: 11
- same_stage_fit_lit: 3
- adjacent_device_name: 56
- periodic_similarity: 56
- ait_excluded: 378

## Quality
- isolated_nodes: 0
- isolated_ratio: 0.000000

## Column Alignment
- missing_in_train_vs_list: 0
- extra_in_train_vs_list: 0

## Model
- recommended_model: Heterogeneous GNN (static graph baseline)