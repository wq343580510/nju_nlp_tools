
from __future__ import print_function

import time
import numpy as np
import dynet
import random
import sys

from ...structure.seg_sentence import SegSentence
from ...structure.seg_sentence import FScore
from Segmenter import Segmenter


class LSTM(object):
    def __init__(self, input_dims, output_dims, model):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model = model

        self.W_i = model.add_parameters(
            (output_dims, input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_i = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.W_f = model.add_parameters(
            (output_dims, input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_f = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.W_c = model.add_parameters(
            (output_dims,input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_c = model.add_parameters(
                (output_dims,),
                init = dynet.ConstInitializer(0)
        )
        self.W_o = model.add_parameters(
            (output_dims, input_dims + output_dims),
            init=dynet.UniformInitializer(0.01),
        )
        self.b_o = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )
        self.c0 = model.add_parameters(
            (output_dims,),
            init=dynet.ConstInitializer(0),
        )

        self.W_params = [self.W_i, self.W_f, self.W_c, self.W_o]
        self.b_params = [self.b_i, self.b_f, self.b_c, self.b_o]
        self.params = self.W_params + self.b_params + [self.c0]

    class State(object):
        def __init__(self, lstm):
            self.lstm = lstm

            self.outputs = []

            self.c = dynet.parameter(self.lstm.c0)
            self.h = dynet.tanh(self.c)

            self.W_i = dynet.parameter(self.lstm.W_i)
            self.b_i = dynet.parameter(self.lstm.b_i)

            self.W_f = dynet.parameter(self.lstm.W_f)
            self.b_f = dynet.parameter(self.lstm.b_f)

            self.W_c = dynet.parameter(self.lstm.W_c)
            self.b_c = dynet.parameter(self.lstm.b_c)

            self.W_o = dynet.parameter(self.lstm.W_o)
            self.b_o = dynet.parameter(self.lstm.b_o)

        def add_input(self, input_vec):

            x = dynet.concatenate([input_vec, self.h])

            i = dynet.logistic(self.W_i * x + self.b_i)
            f = dynet.logistic(self.W_f * x + self.b_f)
            g = dynet.tanh(self.W_c * x + self.b_c)
            o = dynet.logistic(self.W_o * x + self.b_o)

            c = dynet.cmult(f, self.c) + dynet.cmult(i, g)
            h = dynet.cmult(o, dynet.tanh(c))

            self.c = c
            self.h = h
            self.outputs.append(h)

            return self

        def output(self):
            return self.outputs[-1]

    def initial_state(self):
        return LSTM.State(self)


class Network(object):
    def __init__(
            self,
            mode,
            lstm_units,
            hidden_units,
            bigrams_size = None,
            unigrams_size=None,
            words_size=None,
            words_dims =None,
            tag_size=None,
            bigrams_dims=None,
            unigrams_dims=None,
            seg_out=None,
            tag_out=None,
            seg_spans=4,
            tag_spans=3,
            droprate=0
    ):
        self.mode = mode
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.droprate = droprate
        if self.mode == 'seg':
            self.bigrams_size = bigrams_size
            self.bigrams_dims = bigrams_dims
            self.unigrams_dims = unigrams_dims
            self.unigrams_size = unigrams_size
            self.seg_size = seg_out
            self.seg_span = seg_spans

        elif self.mode == 'pos-tagging':
            self.words_size = words_size
            self.words_dims = words_dims
            self.tag_size = tag_size
            self.tag_out = tag_out
            self.tag_spans = tag_spans

        self.model = dynet.Model()
        self.trainer = dynet.AdadeltaTrainer(self.model, eps=1e-7, rho=0.99)
        random.seed(1)

        self.activation = dynet.rectify

        if self.mode is 'pos-tagging':
            self.words_embed = self.model.add_lookup_parameters(
                (self.words_size,self.words_dims)
            )
            self.fwd_lstm1 = LSTM(self.words_size,
                                  self.lstm_units,self.model)
            self.back_lstm1 =LSTM(self.words_size,
                                  self.lstm_units,self.model)
            self.fwd_lstm2 = LSTM(2*self.lstm_units,
                                  self.lstm_units,self.model)
            self.back_lstm2 = LSTM(2*self.lstm_units,
                                   self.lstm_units,self.model)

            self.tag_hidden_W = self.model.add_parameters(
                (self.hidden_units,2 * self.tag_spans * self.lstm_units),
                dynet.UniformInitializer(0.01)
            )
            self.tag_hidden_b = self.model.add_parameters(
                (self.hidden_units,),
                dynet.ConstInitializer(0)
            )
            self.tag_out_W = self.model.add_parameters(
                (self.tag_size,self.hidden_units),
                dynet.UniformInitializer(0.01)
            )
            self.tag_out_b = self.model.add_parameters(
                (self.tag_size,),
                dynet.ConstInitializer(0)
            )


        elif self.mode is 'seg':
            self.bigram_embed = self.model.add_lookup_parameters(
                (self.bigrams_size, self.bigrams_dims),
            )
            self.unigram_embed = self.model.add_lookup_parameters(
                (self.unigrams_size, self.unigrams_dims),
            )
            self.fwd_lstm1 = LSTM(self.bigrams_dims + self.unigrams_dims,
                                 self.lstm_units, self.model)
            self.back_lstm1 = LSTM(self.bigrams_dims + self.unigrams_dims,
                                  self.lstm_units, self.model)

            self.fwd_lstm2 = LSTM(2*self.lstm_units,self.lstm_units,self.model)
            self.back_lstm2 = LSTM(2*self.lstm_units,self.lstm_units,self.model)

            self.seg_hidden_W = self.model.add_parameters(
                (self.hidden_units, 2 * self.seg_span * self.lstm_units),
                dynet.UniformInitializer(0.01)
            )
            self.seg_hidden_b = self.model.add_parameters(
                (self.hidden_units,),
                dynet.ConstInitializer(0)
            )
            self.seg_output_W = self.model.add_parameters(
                (self.seg_size, self.hidden_units),
                dynet.ConstInitializer(0)
            )
            self.seg_output_b = self.model.add_parameters(
                (self.seg_size,),
                dynet.ConstInitializer(0)
            )


    def init_params(self):


        self.bigram_embed.init_from_array(
            np.random.uniform(-0.01, 0.01, self.bigram_embed.shape())
        )
        self.unigram_embed.init_from_array(
            np.random.uniform(-0.01, 0.01, self.unigram_embed.shape())
        )
        self.words_embed.init_from_array(
            np.random.uniform(-0.01,0.01,self.words_embed.shape()),
        )

    def prep_params(self):
        if self.mode == 'seg':
            self.W1_seg = dynet.parameter(self.seg_hidden_W)
            self.b1_seg = dynet.parameter(self.seg_hidden_b)

            self.W2_seg = dynet.parameter(self.seg_output_W)
            self.b2_seg = dynet.parameter(self.seg_output_b)

        elif self.mode == 'pos-tagging':
            self.W1_tag = dynet.parameter(self.tag_hidden_W)
            self.b1_tag = dynet.parameter(self.tag_hidden_b)

            self.W2_tag = dynet.parameter(self.tag_out_W)
            self.b2_tag = dynet.parameter(self.tag_out_b)


    def evaluate_recurrent_tag(self,words,test=False):
        fwd1 = self.fwd_lstm1.initial_state()
        back1 = self.back_lstm1.initial_state()

        fwd2 = self.fwd_lstm2.initial_state()
        back2 = self.back_lstm2.initial_state()

        sentence = []
        for w in words:
            wordvec = dynet.lookup(self.words_embed,w)
            sentence.append(wordvec)

        fwd1_out = []
        for vec in sentence:
            fwd1 = fwd1.add_input(vec)
            fwd_vec = fwd1.output()
            fwd1_out.append(fwd_vec)

        back1_out = []
        for vec in reversed(sentence):
            back1 = back1.add_input(vec)
            back_vec = back1.output()
            back1_out.append(back_vec)

        lstm2_input = []
        for (f,b) in zip(fwd1_out,reversed(back1_out)):
            lstm2_input.append(dynet.concatenate([f,b]))

        fwd2_out = []
        for vec in lstm2_input:
            if self.droprate > 0 and not test :
                vec = dynet.dropout(vec,self.droprate)
            fwd2 = fwd2.add_input(vec)
            fwd_vec = fwd2.output()
            fwd2_out.append(fwd_vec)

        back2_out = []
        for vec in reversed(lstm2_input):
            if self.droprate > 0 and not test:
                vec = dynet.dropout(vec, self.droprate)
            back2 = back2.add_input(vec)
            back_vec = back2.output()
            back2_out.append(back_vec)

        return fwd2_out,back2_out[::-1]

    def evaluate_recurrent_seg(self, fwd_bigrams,unigrams, test=False):
        fwd1 = self.fwd_lstm1.initial_state()
        back1 = self.back_lstm1.initial_state()

        fwd2 = self.fwd_lstm2.initial_state()
        back2 = self.back_lstm2.initial_state()

        fwd_input = []
        for i in range(len(unigrams)):
            bivec = dynet.lookup(self.bigram_embed,fwd_bigrams[i])
            univec = dynet.lookup(self.unigram_embed,unigrams[i])
            vec = dynet.concatenate([bivec,univec])
         #   fwd_input.append(dynet.tanh(self.embed2lstm_W*vec))
            fwd_input.append(vec)
            
        back_input = []
        for i in range(len(unigrams)):
            bivec = dynet.lookup(self.bigram_embed,fwd_bigrams[i+1])
            univec = dynet.lookup(self.unigram_embed,unigrams[i])
            vec = dynet.concatenate([bivec,univec])
           # back_input.append(dynet.tanh(self.embed2lstm_W*vec))
            back_input.append(vec)
            
        fwd1_out = []
        for vec in fwd_input:
            fwd1 = fwd1.add_input(vec)
            fwd_vec = fwd1.output()
            fwd1_out.append(fwd_vec)

        back1_out = []
        for vec in reversed(back_input):
            back = back1.add_input(vec)
            back1_vec = back.output()
            back1_out.append(back1_vec)

        lsmt2_input = []
        for (f,b) in zip(fwd1_out,reversed(back1_out)):
            lsmt2_input.append(dynet.concatenate([f,b]))

        fwd2_out = []
        for vec in lsmt2_input:
            if self.droprate > 0 and not test:
                vec = dynet.dropout(vec,self.droprate)
            fwd2 = fwd2.add_input(vec)
            fwd_vec = fwd2.output()
            fwd2_out.append(fwd_vec)

        back2_out = []
        for vec in reversed(lsmt2_input):
            if self.droprate > 0 and not test:
                vec = dynet.dropout(vec,self.droprate)
            back2 = back2.add_input(vec)
            back_vec = back2.output()
            back2_out.append(back_vec)

        # fwd_out = [dynet.concatenate([f1,f2]) for (f1,f2) in zip(fwd1_out,fwd2_out)]
        # back_out = [dynet.concatenate([b1,b2]) for (b1,b2) in zip(back1_out,back2_out)]

        return fwd2_out,back2_out[::-1]


    def evaluate_segs(self,fwd_out,back_out,lefts,rights,test=False):
        fwd_span_out = []
        for left_index,right_index in zip(lefts,rights):
            fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index-1])
        fwd_span_vec = dynet.concatenate(fwd_span_out)

        back_span_out = []
        for left_index,right_index in zip(lefts,rights):
            back_span_out.append(back_out[left_index] - back_out[right_index+1])
        back_span_vec = dynet.concatenate(back_span_out)

        hidden_input = dynet.concatenate([fwd_span_vec,back_span_vec])

        if self.droprate >0 and not test:
            hidden_input = dynet.dropout(hidden_input,self.droprate)

        hidden_output = self.activation(self.W1_seg * hidden_input + self.b1_seg)

        scores = (self.W2_seg* hidden_output + self.b2_seg)

        return scores

    def evaluate_tags(self,fwd_out,back_out,lefts,rights,test=False):

        fwd_span_out = []
        for left_index, right_index in zip(lefts,rights):
            fwd_span_out.append(fwd_out[right_index] - fwd_out[left_index -1 ])
        fwd_span_vec = dynet.concatenate(fwd_span_out)

        back_span_out = []
        for left_index,right_index in zip(lefts,rights):
            back_span_out.append(back_out[left_index] - back_out[right_index + 1])
        back_span_vec = dynet.concatenate(back_span_out)

        hidden_input = dynet.concatenate([fwd_span_vec,back_span_vec])

        if self.droprate > 0 and not test:
            hidden_input = dynet.dropout(hidden_input,self.droprate)

        hidden_output = self.activation(self.W1_tag * hidden_input + self.b1_tag)

        scores = (self.W2_tag * hidden_output + self.b2_tag)

        return scores
            
        

    def save(self,filename):

        self.model.save(filename)

        with open (filename,'a') as f:
            f.write('\n')
            f.write('bigrams_size = {}\n'.format(self.bigrams_size))
            f.write('unigrams_size = {}\n'.format(self.unigrams_size))
            f.write('bigrams_dims = {}\n'.format(self.bigrams_dims))
            f.write('unigrams_dims = {}\n'.format(self.unigrams_dims))
            f.write('lstm_units = {}\n'.format(self.lstm_units))
            f.write('hidden_units = {}\n'.format(self.hidden_units))
            f.write('lable_size = {}\n'.format(self.label_size))
            f.write('span_nums = {}\n'.format(self.span_nums))



    @staticmethod
    def load(filename):
        """
        Loads file created by save() method
        """

        with open(filename) as f:
            f.readline()
            f.readline()
            bigrams_size = int(f.readline().split()[-1])
            unigrams_size = int(f.readline().split()[-1])
            bigrams_dims = int(f.readline().split()[-1])
            unigrams_dims = int(f.readline().split()[-1])
            lstm_units = int(f.readline().split()[-1])
            hidden_units = int(f.readline().split()[-1])
            label_size = int(f.readline().split()[-1])
            span_nums = int(f.readline().split()[-1])

        network = Network(
            bigrams_size=bigrams_size,
            unigrams_size=unigrams_size,
            bigrams_dims=bigrams_dims,
            unigrams_dims=unigrams_dims,
            lstm_units=lstm_units,
            hidden_units=hidden_units,
            label_size=label_size,
            span_nums=span_nums
        )

        network.model.load(filename)

        return network

    @staticmethod
    def train(
        corpus,
        bigrams_dims,
        unigrams_dims,
        lstm_units,
        hidden_units,
        epochs,
        batch_size,
        train_data_file,
        dev_data_file,
        model_save_file,
        droprate,
        unk_params,
        alpha,
        beta
    ):

        start_time = time.time()

        fm = corpus
        bigrams_size = corpus.total_bigrams()
        unigrams_size = corpus.total_unigrams()

        network = Network(
            bigrams_size=bigrams_size,
            unigrams_size=unigrams_size,
            bigrams_dims=bigrams_dims,
            unigrams_dims=unigrams_dims,
            lstm_units = lstm_units,
            hidden_units = hidden_units,
            label_size = fm.total_labels(),
            span_nums = fm.total_span_nums(),
            droprate=droprate,
        )

        network.init_params()


        print('Hidden units : {} ,per LSTM units : {}'.format(
            hidden_units,
            lstm_units,
        ))

        print('Embeddings: bigrams = {}, unigrams = {}'.format(
            (bigrams_size,bigrams_dims),
            (unigrams_size,unigrams_dims)
        ))

        print('Dropout rate : {}'.format(droprate))
        print('Parameters initialized in [-0.01,0.01]')
        print ('Random UNKing parameter z = {}'.format(unk_params))

        training_data = corpus.gold_data_from_file(train_data_file)
        num_batched = -(-len(training_data) // batch_size)
        print('Loaded {} training sentences ({} batches of size {})!'.format(
            len(training_data),
            num_batched,
            batch_size,)
        )

        parse_every = -(-num_batched // 4)

        dev_sentences = SegSentence.load_sentence_file(dev_data_file)
        print('Loaded {} validation sentences!'.format(len(dev_sentences)))


        best_acc =FScore()
        for epoch in xrange(1,epochs + 1):
            print ('............ epoch {} ............'.format(epoch))

            total_cost = 0.0
            total_states = 0
            training_acc = FScore()

            #np.random.shuffle(training_data)

            for b in xrange(num_batched):
                batch = training_data[(b * batch_size) : (b + 1) * batch_size]

                explore = [
                    Segmenter.exploration(
                        example,
                        fm,
                        network,
                        alpha=alpha,
                        beta=beta
                    ) for example in batch
                ]
                for (_,acc ) in explore:
                    training_acc += acc

                batch = [example for (example, _ ) in explore]

                dynet.renew_cg()
                network.prep_params()

                errors = []
                for example in batch:
                    ## random UNKing ##
                    for (i,uni) in enumerate(example['unigrams']):
                        if uni <= 2:
                            continue

                        u_freq = fm.unigrams_freq_list[uni]
                        drop_prob = unk_params / (unk_params + u_freq)
                        r = np.random.random()
                        if r < drop_prob :
                            example['unigrams'][i] = 0

                    for (i,bi) in enumerate(example['fwd_bigrams']):
                        if bi <= 2:
                            continue

                        b_freq = fm.bigrams_freq_list[bi]
                        drop_prob = unk_params / (unk_params + b_freq)
                        r = np.random.random()
                        if r < drop_prob:
                            example['fwd_bigrams'][i] = 0


                    fwd,back = network.evaluate_recurrent(
                        example['fwd_bigrams'],
                        example['unigrams'],
                    )

                    for (left,right),correct in example['seg_data'].items():
                        # correct = example['label_data'][(left,right)]
                        scores = network.evaluate_labels(fwd,back,left,right)

                        probs = dynet.softmax(scores)
                        loss = -dynet.log(dynet.pick(probs,correct))
                        errors.append(loss)
                    total_states += len(example['seg_data'])

                    for (left,right),correct in example['tag_data'].items():
                        scores = network.evaluate_tags(fwd,back,left,right)

                        probs = dynet.softmax(scores)
                        loss = -dynet.log(dynet.pick(probs,correct))
                        errors.append(loss)
                    total_states += len(example['tag_data'])


                batch_error = dynet.esum(errors)
                total_cost += batch_error.scalar_value()
                batch_error.backward()
                network.trainer.update()

                mean_cost = total_cost/total_states

                print(
                    '\rBatch {}  Mean Cost {:.4f}  [Train: {}]'.format(
                        b,
                        mean_cost,
                        training_acc,
                    ),
                    end='',
                )
                sys.stdout.flush()

                if ((b+1) % parse_every) == 0 or b == (num_batched - 1):
                    dev_acc = Segmenter.evaluate_corpus(
                        dev_sentences,
                        fm,
                        network,
                    )
                    print (' [Val: {}]'.format(dev_acc))

                    if dev_acc.fscore() >best_acc.fscore():
                        best_acc = dev_acc
                        network.save(model_save_file)
                        print('    [saved model : {}]'.format(model_save_file))

            current_time = time.time()
            runmins = (current_time - start_time)/60
            print(' Elapsed time: {:.2f}m'.format(runmins))

        return network
