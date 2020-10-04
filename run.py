from __future__ import print_function
import sugartensor as tf
import numpy as np
from prepro import *
from train import ModelGraph
import codecs
import readchar

def main(): 
    g = ModelGraph(mode="test")
        
    with tf.Session() as sess:
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train'))
        print("Restored!")
        mname = open('asset/train/checkpoint', 'r').read().split('"')[1] # model name

        char2idx, idx2char = load_char_vocab()
        word2idx, idx2word = load_word_vocab()


        previous = [0]*50 # a stack for previous words
        para = "EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
        ctx = [0]*50

        while True:
            key = readchar.readkey().lower()

            if key == readchar.key.BACKSPACE:
                ctx.insert(0, previous.pop())
                ctx.pop()
                previous.insert(0, 0)

            elif key == readchar.key.ESC:
                break

            else:
                key_idx = char2idx[key]
                ctx.append(key_idx)
                ctx.pop(0)

            logits = sess.run(g.logits, {g.x: np.expand_dims(ctx, 0)})
            preds = logits.argsort()[0][-3:]
            # pred = np.argmax(logits, -1)[0]
            predword1, predword2, predword3 = [idx2word.get(pred) for pred in preds]
            print(predword1, ' ', predword2, ' ', predword3)


                                        
if __name__ == '__main__':
    main()
    print("Done")

