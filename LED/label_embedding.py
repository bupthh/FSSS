from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class LabelEmbedding(object):

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)
    print("module %s loaded" % module_url)

    @staticmethod
    def build_embedding_file(label_set, file_url):
        message_embeddings = LabelEmbedding.model(label_set)

        with open(file_url, 'w') as ef:
            for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
                message_embedding_snippet = " ".join((str(x) for x in message_embedding[:]))
                print("{}".format(message_embedding_snippet), file=ef)


if __name__ == '__main__':
    bird_labels = ['cock', 'hen', 'ratite, ratite bird, flightless bird', 'ostrich, Struthio camelus']
    LabelEmbedding.build_embedding_file(bird_labels, 'd:/bird_labels.embed')
