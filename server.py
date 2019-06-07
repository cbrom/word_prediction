# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
import codecs
import readchar
import simplejson as json
from flask import Flask, render_template,request

import sugartensor as tf
import numpy as np
from prepro import *
from train import ModelGraph

app = Flask(__name__)


@app.route("/test")
def output():
      return render_template("index.html")


@app.route('/output', methods=['GET'])
def worker():
    #print(request, file=sys.stderr)
    string = request.args.get('string').lower()
    work = request.args.get('work')
    words=string.split()
    #print(words, file=sys.stderr)
    n=len(words)

    latest_50_chars = string[-50:]
    para = "E"*(50 - len(latest_50_chars)) + latest_50_chars
    ctx = [char2idx[char] for char in para]

    logits = sess.run(g.logits, {g.x: np.expand_dims(ctx, 0)})
    preds = logits.argsort()[0][-3:]

    predword1, predword2, predword3 = [idx2word.get(pred) for pred in preds]

    return json.dumps([(predword1, ), (predword2, ), (predword3, )])

     
if __name__=="__main__":
    g = ModelGraph(mode="test")

    with tf.Session() as sess:
        tf.sg_init(sess)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
        print('Restored')

        char2idx, idx2char = load_char_vocab()
        word2idx, idx2word = load_word_vocab()

        previous = [0]*50 # a stack for previous words
        para = "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
        ctx = [0]*50


        app.run(debug=True)
