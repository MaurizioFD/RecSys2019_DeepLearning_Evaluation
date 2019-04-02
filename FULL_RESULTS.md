# DeepLearning RS Evaluation

## Compilation and experiments
Instructions on how to install and run the experiments are accessible [HERE](README.md).


## Full results
http://pandoc.org/try/


* [SIGIR: Collaborative Memory Networks](#SIGIR-Collaborative-Memory-Networks)
* [RecSys: Spectral Collaborative Filtering](#RecSys-Spectral-Collaborative-Filtering)
* [KDD: Leveraging Meta-path based Context for Top-N Recommendation with a NeuralCo-Attention Model](#KDD-Leveraging-Meta-path-based-Context-for-Top-N-Recommendation-with-a-NeuralCo-Attention-Model)
* [KDD: Collaborative Variational Autoencoder for Recommender Systems](#KDD-Collaborative-Variational-Autoencoder-for-Recommender-Systems)
* [KDD: Collaborative Deep Learning](#KDD-Collaborative-Deep-Learning)
* [WWW: Neural Collaborative Filtering](#WWW-Neural-Collaborative-Filtering)
* [WWW: Variational Autoencoders for Collaborative Filtering](#WWW-Variational-Autoencoders-for-Collaborative-Filtering)






### SIGIR: Collaborative Memory Networks

### RecSys: Spectral Collaborative Filtering

### KDD: Leveraging Meta-path based Context for Top-N Recommendation with a NeuralCo-Attention Model

### KDD: Collaborative Variational Autoencoder for Recommender Systems

### KDD: Collaborative Deep Learning

### WWW: Neural Collaborative Filtering



### WWW: Variational Autoencoders for Collaborative Filtering

| Movielens 20M |   REC@20   |  NDCG@20   |   REC@50   |  NDCG@50   |  REC@100   |  NDCG@100  |
| :--------- | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| Random     |   0.0011   |   0.0009   |   0.0027   |   0.0016   |   0.0050   |   0.0024   |
| TopPop     |   0.1441   |   0.1201   |   0.2320   |   0.1569   |   0.3296   |   0.1901   |
| ItemKNN CF |   0.2851   |   0.2428   |   0.4328   |   0.3030   |   0.5547   |   0.3459   |
| P3alpha    |   0.2607   |   0.2157   |   0.4053   |   0.2739   |   0.5296   |   0.3179   |
| RP3beta    |   0.2949   |   0.2454   |   0.4411   |   0.3054   |   0.5694   |   0.3507   |
| MultiVAE   | **0.3540** | **0.2993** | **0.5220** | **0.3693** | **0.6488** | **0.4154** |


| Netflix Prize |   REC@20   |  NDCG@20   |   REC@50   |  NDCG@50   |  REC@100   |  NDCG@100  |
| :--------- | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| Random     |   0.0011   |   0.0011   |   0.0031   |   0.0020   |   0.0058   |   0.0031   |
| TopPop     |   0.0782   |   0.0761   |   0.1643   |   0.1159   |   0.2718   |   0.1570   |
| ItemKNN CF |   0.2088   |   0.1966   |   0.3386   |   0.2588   |   0.4590   |   0.3086   |
| P3alpha    |   0.1977   |   0.1762   |   0.3346   |   0.2414   |   0.4675   |   0.2967   |
| RP3beta    |   0.2196   |   0.2044   |   0.3560   |   0.2699   |   0.4886   |   0.3246   |
| MultiVAE   | **0.2626** | **0.2448** | **0.4138** | **0.3192** | **0.5476** | **0.3756** |

Experimental results for MultiVAE on the metrics and cutoffs reported in
the original paper.



| Algorithm | Parameter            | movielens20m | netflixPrize |  
| :-------- | :------------------- | :----------: | :----------: |  
| ItemKNN CF          | topK                 |     312      |     160      |  
|           | shrink               |      10      |     965      |  
|           | similarity           |    cosine    |    cosine    |  
|           | normalize            |     True     |     True     |  
|           | feature weighting    |    TF-IDF    |     BM25     |  
| P3alpha          | topK                 |     533      |     684      |  
|           | alpha                |    0.4313    |    2.0000    |  
|           | normalize similarity |     True     |     True     |  
| RP3beta          | topK                 |     348      |     290      |  
|           | alpha                |    1.2252    |    0.8838    |  
|           | beta                 |    0.3832    |    0.5335    |  
|           | normalize similarity |     True     |     True     |  
| MultiVAE          | epochs               |      65      |      60      |  
|           | batch size           |     500      |     500      |  
|           | total anneal steps   |    200000    |    200000    |  
|           | p dims               |      \-      |      \-      |  

Parameter values for our baselines on all reported algorithms and
datasets.