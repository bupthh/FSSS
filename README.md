We provide an open source Python package of the feature selection method based on our paper "Feature Selection for Hierarchical Classification via Joint Semantic and Structural Information of Labels". In this paper, we propose a novel Feature Selection based on Semantic and Structural information of labels (FSSS) framework. First, we transform the label description to a mathematical representation and calculate the similarity score between labels as the semantic regularization. Second, we investigate the hierarchical relations in a tree structure of the label space as the structural regularization. Finally, we impose the two regularization terms on a sparse learning based model to jointly guide the feature selection. 

In this package, two programs are provided as following:

<b>FSSS</b>: This is our FSSS framework of feature selection, which includes training data building, algorithm implementation of feature selection, and output data generation based on selected features.  

<b>LDB</b>: This is for the generation of Label Description Embedding (LDB). We use the Universal Sentence Encoder (https://tfhub.dev/google/collections/universal-sentence-encoder/1) to transform the label text. This encoder uses a transformer-network that is trained on a variety of data sources and a variety of tasks. By inputting a variable length label text, we can get the output of a 512-dimensional vector.


Prerequisites:

Python 3.x

tensorflow

tensorflow-hub

