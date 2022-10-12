optimizer : {
  type: AdamW,
  kwargs: {
  lr : 0.001,
  weight_decay : 0.05
}}

scheduler: {
  type: StepLR,  #CosLR
  kwargs: {
    epochs: 200, #300
    initial_epochs : 1  #10
}}

dataset : {
  train : { _base_: cfgs/dataset_configs/ShapeNet-55.yaml,
            others: {subset: 'train', npoints: 1024}},
  val : { _base_: cfgs/dataset_configs/ShapeNet-55.yaml,
            others: {subset: 'test', npoints: 1024}},
  test : { _base_: cfgs/dataset_configs/ShapeNet-55.yaml,
            others: {subset: 'test', npoints: 1024}}}

model : {
  NAME: Point_MAE, #PointTransformer
  group_size: 32,
  num_group: 64,
  loss: cdl2,
  transformer_config: {
    mask_ratio: 0.6,
    mask_type: 'rand',
    trans_dim: 384,
    encoder_dims: 384,
    depth: 12,
    drop_path_rate: 0.1,
    num_heads: 6,
    decoder_depth: 4,
    decoder_num_heads: 6,
  },
  }

npoints: 1024 # hardeset 2048
total_bs : 32 #128 #4
step_per_update : 1
max_epoch : 100 #300 #1
