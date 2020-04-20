import numpy as np

class Saver(object):
    def save(self, model, sess):
        print('save no implemented')

class MFSaver(Saver):
    def __init__(self):
        self.prefix = None

    def setPrefix(self, prefix = None):
        self.prefix = prefix

    def save(self, model, sess):
        if self.prefix == None:
            print ("prefix should be set by GMFSaver.setPrefix(prefix)")
            return

        params = sess.run([model.embedding_P, model.embedding_Q])
        print ('saving model.embedding_P', params[0].shape, ', model.embedding_Q', params[1].shape,\
              ' to', self.prefix, "_*.txt")

        f = open(self.prefix + "_P.txt", 'w')
        np.savetxt(f, params[0])
        f.close()

        f = open(self.prefix + "_Q.txt", 'w')
        np.savetxt(f, params[1])
        f.close()


class GMFSaver(Saver):
    def __init__(self):
        self.prefix = None

    def setPrefix(self, prefix = None):
        self.prefix = prefix