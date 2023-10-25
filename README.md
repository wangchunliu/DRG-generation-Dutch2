
This repository is based on the [code](https://github.com/UKPLab/emnlp2019-dualgraph).
Different from their work, this repository focuses on the DRG-to-text generation task. 
[Another repository](https://github.com/wangchunliu/DRG-generation-Dutch) is its sister, which includes the poster for CLIN 32 conference.

## Environments and Dependencies

- Python 3.6
- PyTorch 1.5.0

## Datasets

In our experiments, we use the following datasets same as [sister repositoty](https://github.com/wangchunliu/DRG-generation-Dutch) :

## Preprocess

First, convert the dataset into the format required for the model.

```
sh preprocess_nl.sh   
```


## Training
For training the model using the DRG dataset, execute:

```
./train_sh.sh <gpu_id> <gnn_type> <gnn_layers> <start_decay_steps> <decay_steps>
```
Options for `<gnn_type>` are `ggnn`, `gat` or `gin`. `<gnn_layers>` is the number of graph layers. Refer to [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) for `<start_decay_steps>` and `<decay_steps>`.

We lower the learning rate during training, after some epochs, as in Konstas et al. (2017).

Examples:
```
sh train_nl.sh 0 gin 2 5000 5000
```

## Decoding

For decode on the test set, run:
```
sh decode_nl.sh <gpu_id> <model> <nodes_file> <node1_file> <node2_file> <output>
```

Example:
```
sh decode_nl.sh 0 model_ggnn.pt test.txt-src-nodes.txt test.txt-src-node1.txt test.txt-src-node2.txt output-ggnn-test.txt
```


## Cite
```
@article{wang-2022-dutch-compare,
  title={Comparing Neural Meaning-to-Text Approaches for Dutch},
  author={Wang, Chunliu and
      Bos, Johan},
  journal={Computational Linguistics in the Netherlands Journal},
  volume={12},
  pages={269–286},
  year={2022}
}
```
