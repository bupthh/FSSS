from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class LabelEmbedding(object):
    """
    This class is for building the file of embedding representation vectors for all label descriptions.
    we use the Universal Sentence Encoder (https://tfhub.dev/google/collections/universal-sentence-encoder/1) to transform the label text.
    This encoder uses a transformer-network that is trained on a variety of data sources and a variety of tasks.
    By inputting a variable length label text, we can get the output of a 512-dimensional vector.
    """

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)
    print("module %s loaded" % module_url)

    @staticmethod
    def build_embedding_file(label_set, file_url):
        """
        build the file of embedding representation vectors for all label descriptions.
        :param label_set: the list for all labels
        :param file_url:  the output file to record  embedded vectors for all labels.
        :return: None
        """
        message_embeddings = LabelEmbedding.model(label_set)

        with open(file_url, 'w') as ef:
            for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
                message_embedding_snippet = " ".join((str(x) for x in message_embedding[:]))
                print("{}".format(message_embedding_snippet), file=ef)


if __name__ == '__main__':
    bird_labels = ['cock', 'hen', 'ratite, ratite bird, flightless bird', 'ostrich, Struthio camelus']
    LabelEmbedding.build_embedding_file(bird_labels, 'bird_labels.embed')
