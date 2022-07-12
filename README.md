# 2022-KGAT
Reimplementation of the KGAT with Pytorch and DGL(Deep Graph Library) 
## Knowledge Graph Attention Network
>Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. [Paper in ACM DL](https://dl.acm.org/authorize.cfm?key=N688414) or [Paper in arXiv](https://arxiv.org/abs/1905.07854). In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.

You can find Tensorflow implementation by the paper authors [here](https://github.com/xiangwang1223/knowledge_graph_attention_network) and other pytorch version [here](https://github.com/LunaBlack/KGAT-pytorch)


## Introduction
Knowledge Graph Attention Network (KGAT) is a study recommended the introduction of task knowledge profiles and user behavior data, to explore the higher-order information as side information, thereby enhancing the user to predict based on the user's interaction with an item A question of preference to provide more accurate, more diverse and easier to explain recommendations.

## Citation
If you want to use codes and datasets in your research, please cite:
```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat{-}Seng Chua},
  title     = {{KGAT:} Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  pages     = {950--958},
  year      = {2019}
}
```

## Prerequisites

Create environment with python 3.8

```shell script
conda create -n kgat python=3.8.10
```

Install requirements:
```shell script
conda activate kgat
conda install cudatoolkit=11.3
pip install -r requirements.txt
pip install dgl-cu113==0.8.1 dglgo -f https://data.dgl.ai/wheels/repo.html
```

List of required packages:
* CUDA == 11.3
* torch == 1.11.0
* numpy == 1.22.4
* pandas == 1.4.2
* scipy == 1.8.1
* tqdm == 4.64.0
* scikit-learn == 1.1.1
* dgl-cu113 = 0.8.1

## Run the Codes
### From source code
```
python Main.py --data_name last-fm
```
### From dockerfile
```
docker pull nvidia/cuda:11.3.0-runtime-ubuntu20.04
# the same dir with dockerfile
docker build -t cuda_2022_kgat:demo .
docker run -it --gpus all --name KGAT_container cuda_2022_kgat:demo
```


## Dataset
We provide three processed datasets: Amazon-book, Last-FM, and Yelp2018.
* You can find the full version of recommendation datasets via [Amazon-book](http://jmcauley.ucsd.edu/data/amazon), [Last-FM](http://www.cp.jku.at/datasets/LFM-1b/), and [Yelp2018](https://www.yelp.com/dataset/challenge).
* We follow [KB4Rec](https://github.com/RUCDM/KB4Rec) to preprocess Amazon-book and Last-FM datasets, mapping items into Freebase entities via title matching if there is a mapping available.

|                       |               | Amazon-book |   Last-FM |  Yelp2018 |
| :-------------------: | :------------ | ----------: | --------: | --------: |
| User-Item Interaction | #Users        |      70,679 |    23,566 |    45,919 |
|                       | #Items        |      24,915 |    48,123 |    45,538 |
|                       | #Interactions |     847,733 | 3,034,796 | 1,185,068 |
|    Knowledge Graph    | #Entities     |      88,572 |    58,266 |    90,961 |
|                       | #Relations    |          39 |         9 |        42 |
|                       | #Triplets     |   2,557,746 |   464,567 | 1,853,704 |

* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: (`userID` and `a list of itemID`).
  
* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: (`userID` and `a list of itemID`).
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.
  
* `final_kg.txt`
  * Knowledge graph file.
  * Each line is a triplet (`head_id`, `relation_id`, `tail_id`) for two entity and one relation in knowledge graph, where `head_id` and `tail_id` represent the ID of such entities with relationship `relation_id`.

## Acknowledgement
Any scientific publications that use this datasets should cite the following paper as the reference:
```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat-Seng Chua},
  title     = {KGAT: Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  year      = {2019}
}
```