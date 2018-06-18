import math
import pickle
import numpy as np
import argparse
import msgpack
from random import randint, shuffle
from numpy import array
from numpy import argmax
from numpy import array_equal
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.optimizers import Nadam
from attention_decoder import AttentionDecoder
from gensim.models import Word2Vec
from keras.objectives import cosine_proximity
from keras import backend as K

parser = argparse.ArgumentParser(description='Encoder-Decoder')
parser.add_argument('--lr', type=float, default=0.005, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.002, metavar='N',
                    help='L2 weight decay')
parser.add_argument('--batch_size', type=int, default=5000, metavar='N',
                    help='global batch size')
parser.add_argument('--num_epochs', type=int, default=10, metavar='N',
                    help='maximum number of epochs')
parser.add_argument('--timesteps_in', type=int, default=80, metavar='N',
                    help='maximum number of time steps of Encoder')
parser.add_argument('--timesteps_out', type=int, default=80, metavar='N',
                    help='maximum number of time steps of Decoder')
parser.add_argument('--optimizer', type=str, default="nadam", metavar='N',
                    help='optimizer kind: nadam, momentum, adagrad or rmsprop')
parser.add_argument('--hidden_layers', type=str, default="256,256", metavar='N',
                    help='hidden layer sizes, comma-separated')
parser.add_argument('--path_to_word2vec', type=str, default='../v2/annoy-dim-256-tree-100-mincount-10'
                    , metavar='N', help='Path to word2vec model')
parser.add_argument('--path_to_train_data', type=str, default='../data-1000k.msgpack', metavar='N',
                    help='Path to training data')
parser.add_argument('--non_linearity_type', type=str, default="selu", metavar='N',
                    help='type of the non-linearity used in activations')
parser.add_argument('--vector_dim', type=int, default=256, metavar='N',
                    help='size of song embedding')


args = parser.parse_args()
print(args)

def get_word2vec(embedding_dim, path=''):
    directory = path
    w2v_fname = directory + '/word2vec-' + str(embedding_dim) + '.txt'
    print('Loading Word2Vec Model...')
    model = Word2Vec.load(w2v_fname)
    print('Loaded Word2Vec Model...')
    return model


# Load data from file
def get_data(length=1000000, data_path=""):
    print('Loading Data')
    with open(data_path, 'rb') as f:
        train_data_layer = msgpack.load(f, encoding='utf-8')
    print('Data Loaded!')
    return train_data_layer[:length]
    


class Load_data():
    def __init__(self, data, w2v_model, n_in, n_out, vec_dim, batch_size):
        self.data = data
        self.w2v_model = w2v_model
        self.n_in = n_in
        self.n_out = n_out
        self.vec_dim = vec_dim
        self.batch_size = batch_size
                
    # prepare data for the LSTM
    def get_pair_all(self, data, n_in, n_out, vec_dim):
        # generate random sequence
        ip = []
        op = []
        random_shape = self.w2v_model.wv.vectors[0].shape[0]
        for pl in range(len(data)):
            X = data[pl][:n_in]
            newX = []
            for x in X:
                if x not in self.w2v_model.wv.vocab:
                    newX.append(np.random.rand(random_shape))
                    print("Song ID: {} not found!".format(x))
                else:
                    newX.append(self.w2v_model.wv[x])
            for fill in range(n_in - len(X)):
                newX.append(np.zeros_like(self.w2v_model.wv.vectors[0], dtype=float))

            X = newX
            newX = []
            Y = X[:n_out]
            #Make array out of them
            X = np.array(X)
            Y = np.array(Y)
            # reshape as 3D
            X = X.reshape((n_in, vec_dim))
            Y = Y.reshape((n_out, vec_dim))
            ip.append(X)
            op.append(Y)
        return ip, op

    #Get data in batches
    def generate_batch_data(self, is_shuffle=True):
        data = self.data
        if is_shuffle:
            shuffle(data)
            
        s_ind = 0
        data_size = len(data)
        e_ind = min(self.batch_size, data_size)
        while e_ind <= data_size:
            if (data_size - e_ind) < self.batch_size:
                (X,y) = self.get_pair_all(data[s_ind:len(data)], self.n_in, self.n_out, self.vec_dim)
            else:
                (X,y) = self.get_pair_all(data[s_ind:e_ind], self.n_in, self.n_out, self.vec_dim)
            s_ind += self.batch_size
            e_ind += self.batch_size
            yield  (array(X),array(y))
            del X,y

def cosine_sim(predicted, expected):
    total = 0.0
    for x,y in zip(predicted, expected):
        temp = cos_distance(x,y)
        temp = K.eval(temp)
        total += temp
    return total

def cos_distance(y_true, y_pred):
    def l2_normalize(x, axis):
        norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
        return K.sign(x) * K.maximum(K.abs(x), K.epsilon()) / K.maximum(norm, K.epsilon())
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return K.mean(1 - K.sum((y_true * y_pred), axis=-1))


#Class to create the model        
class EncoDeco():
    def __init__(self, n_timesteps_in, n_features, epochs, batch_size, attention=False):
        self.n_timesteps_in = n_timesteps_in
        self.n_features = n_features
        self. n_timesteps_out = n_timesteps_out
        self.batch_size = batch_size
        self.epochs = epochs
        if attention == True:
            self.model = self.attention_model()
        else:
            self.model = self.baseline_model()

    # define the encoder-decoder model
    def baseline_model(self):
        model = Sequential()
        model.add(LSTM(256, input_shape=(self.n_timesteps_in, self.n_features), return_sequences=True, go_backwards=True))
        model.add(LSTM(256, input_shape=(self.n_timesteps_in, self.n_features), return_sequences=True))
        model.add(LSTM(256, input_shape=(self.n_timesteps_in, self.n_features), return_sequences=False))
        model.add(RepeatVector(self.n_timesteps_in))
        model.add(LSTM(256, return_sequences=True))
        model.add(TimeDistributed(Dense(self.n_features, activation='tanh')))
        nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss=cos_distance, optimizer=nadam, metrics=['acc'])
        return model

    # define the encoder-decoder with attention model
    def attention_model(self):
        model = Sequential()
        model.add(LSTM(150, input_shape=(self.n_timesteps_in, self.n_features), return_sequences=True))
        model.add(AttentionDecoder(150, self.n_features))
        nadam = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss=cos_distance, optimizer=nadam, metrics=['acc'])    
        return model

    # train and evaluate a model, return accuracy
    def train_evaluate_model(self, train_data):
        # train LSTM
        no_of_batches = math.ceil(len(train_data.data)/batch_size)
        for e in range(self.epochs):
            for counter,(X,y) in enumerate(train_data.generate_batch_data()):
                self.model.fit(X, y, epochs=1, verbose=1, batch_size=256, validation_split=0.0001)
                print('Processing Epoch: {}/{}, Batch: {}/{}'.format(e, self.epochs, counter, no_of_batches))
                del X,y


# configure problem
params={}
epochs = params['num_epochs'] = args.num_epochs
n_features = params['vector_dim'] = args.vector_dim                 #Word Embedding Dimension
n_timesteps_in = params['timesteps_in'] = args.timesteps_in         #75th Quartile range
n_timesteps_out = params['timesteps_out'] = args.timesteps_out      # Same size as input
n_repeats = 1
batch_size = params['batch_size'] = args.batch_size                 #One Million Entries, so batchsize=5000 atleast
word2vec_path = params['path_to_word2vec'] = args.path_to_word2vec  #path to word2vec model
data_path = params['path_to_train_data'] = args.path_to_train_data
data = get_data(data_path=data_path)
w2v_model = get_word2vec(n_features, word2vec_path)
train_data = Load_data(data, w2v_model, n_timesteps_in, n_timesteps_out, n_features, batch_size)
del data
del w2v_model
# evaluate encoder-decoder model
print('Training Encoder-Decoder Model')
enc_dec_model = EncoDeco(n_timesteps_in, n_features, epochs, batch_size, attention=False)
results = list()
for _ in range(n_repeats):
    print(enc_dec_model.model.summary())
    enc_dec_model.train_evaluate_model(train_data)
    

print('Saving Model...')
enc_dec_model.model.save('enc_dec_model.h5')
print('Done!')


# Getting Playlist representations
print('Getting 3rd layer output...')
playlist = {}
get_3rd_layer_output = K.function([enc_dec_model.model.layers[0].input],
                                  [enc_dec_model.model.layers[3].output])
for d,(X,y) in enumerate(train_data.generate_batch_data(is_shuffle=False)):
    layer_output = get_3rd_layer_output([X])

layer_output = layer_output[0]
for i in range(X.shape[0]):
    playlist[i] = layer_output[i][0]

print('Saving Playlist Embeddings')
with open('enc_dec_playlist_emb.pkl', 'wb') as fp:
    pickle.dump(playlist, fp)

print('Done!')
        

'''
# evaluate encoder-decoder with attention model
print('Training Encoder-Decoder With Attention Model')
results = list()
attn_model = EncoDeco(n_timesteps_in, n_features, epochs, batch_size, attention=True)
for _ in range(n_repeats):
    accuracy = attn_model.train_evaluate_model(train_data)
    results.append(accuracy)
    print(accuracy)
print('Mean Accuracy: %.2f%%' % (sum(results)/float(n_repeats)))
'''