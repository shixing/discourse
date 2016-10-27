# 00-01 dev; 02-20 train; 21-22 test

import os
import sys
from env import *
from nltk import word_tokenize
import locale

def split():
    f_train = open(os.path.join(data_dir,"train.txt"),'w')
    f_dev = open(os.path.join(data_dir,"dev.txt"),'w')
    f_test = open(os.path.join(data_dir,"test.txt"),'w')
    
    def extract(f_target,folder_list):
        print "collect data from {} to {}".format(folder_list, f_target.name)
        for d in folder_list:
            d = os.path.join(data_dir,"orig/{}".format(d))
            for fn in os.listdir(d):
                if fn.endswith("pipe"):
                    f = open(os.path.join(d,fn))
                    for line in f:
                        f_target.write(line)
                    f.close()

    def digit2_list(start,end):
        l = []
        for i in xrange(start,end+1):

            s = str(i)
            if len(s) == 1:
                s = "0" + s
            l.append(s)
        return l
        
    extract(f_dev, ["00","01"])
    extract(f_test, ["21","22"])
    extract(f_train, digit2_list(2,20))

    f_train.close()
    f_dev.close()
    f_test.close()

class Record:
    def __init__(self):
        self.arg1 = ""
        self.arg2 = ""
        self.type = ""
        self.sense = ""
        self.top_sense = ""
        self.conn = ""

    def parse(self,line):
        ll = line.strip().split("|")
        self.arg1 = ll[24]
        self.arg2 = ll[34]
        self.type = ll[0]
        self.sense = ll[11]
        self.conn = ll[8]
        if self.sense != "":
            self.top_sense = self.sense.split(".")[0]

def key_count(item_list,key_func, filter_func = None):
    if filter_func == None:
        filter_func = lambda x: True
    d = {}
    for item in item_list:
        if not filter_func(item):
            continue
        key = key_func(item)
        if key not in d:
            d[key] = 0
        d[key] += 1
    return d

def print_count(d):
    s = 0
    for key in d:
        s += d[key]
    for key in d:
        print key, d[key], "{:2f}".format(1.0*d[key]/s)
    print "Sum:", s

def load_records(fn, filter_func = lambda x : True):
    f = open(fn)
    records = []
    for line in f:
        r = Record()
        r.parse(line)
        if filter_func(r):
            records.append(r)
    f.close()
    return records

def tokenize_data(folder):
    folder = os.path.join(data_dir, folder)
    print folder
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    def _tokenize(line):
        words = line.strip()
        if words == "":
            return ""
        words = word_tokenize(words)
        words = [x.lower() for x in words]
        # replace number to "N"
        new_words = []
        for word in words:
            try:
                locale.atof(word)
                new_words.append("N")
            except:
                new_words.append(word)
        return " ".join(new_words)

    def _process(fn,dest_fn):
        records = load_records(fn, lambda x: x.type == "Implicit" )
        f1 = open(dest_fn + ".arg1",'w')
        f2 = open(dest_fn + ".arg2",'w')
        fsense = open(dest_fn + ".rl",'w')
        for r in records:
            f1.write(_tokenize(r.arg1)+"\n")
            f2.write(_tokenize(r.arg2)+"\n")
            fsense.write(r.top_sense + "\n")
        f1.close()
        f2.close()
        fsense.close()
        
    _process(train_path, os.path.join(folder,'train'))
    _process(dev_path, os.path.join(folder,'dev'))
    _process(test_path, os.path.join(folder,'test'))


def human_readable():
    def _convert(fn):
        records = load_records(fn, lambda x: x.type == "Explicit")
        f = open(fn + ".hr",'w')
        for r in records:
            f.write(r.arg1 + "\n")
            f.write(r.arg2 + "\n")
            f.write(r.conn + "\n")
            f.write(r.sense + "\n")
            f.write("\n")
        f.close()

    for fn in [train_path,dev_path,test_path]:
        _convert(fn)

def statistics():
    def _stat(fn):
        records = load_records(fn)
        d_type = key_count(records, lambda x:x.type)
        d_top_sense = key_count(records, lambda x:x.top_sense)
        d_ts_explicit = key_count(records, lambda x:x.top_sense, lambda x : x.type == "Explicit")
        d_ts_implicit = key_count(records, lambda x:x.top_sense, lambda x : x.type == "Implicit")
        d_ts_altlex = key_count(records, lambda x:x.top_sense, lambda x : x.type == "AltLex")
        # print out
        print fn
        print "d_type"
        print_count(d_type)
        print 
        print "d_top_sense"
        print_count(d_top_sense)
        print
        print "d_ts_explicit"
        print_count(d_ts_explicit)
        print
        print "d_ts_implicit"
        print_count(d_ts_implicit)
        print
        print "d_ts_altlex"
        print_count(d_ts_altlex)
        print

    _stat(train_path)
    _stat(dev_path)
    _stat(test_path)

def extract_paper():
    def _extract(fn,dest_path, nway=5):
        f = open(fn)
        f1 = open(dest_path+".arg1",'w')
        f2 = open(dest_path+".arg2",'w')
        frl = open(dest_path+".rl",'w')

        sents = []
        rls = []
        for line in f:
            if line.startswith("=== "):
                # process buff
                for i in xrange(len(rls)-1):
                    rl = rls[i]
                    if nway == 4 and rl == -1:
                        continue
                    arg1 = sents[i]
                    arg2 = sents[i+1]
                    f1.write(arg1+"\n")
                    f2.write(arg2+"\n")
                    frl.write(str(rl)+"\n")
                sents = []
                rls = []
            else:
                ll = line.split("\t")
                sents.append(ll[0])
                rls.append(int(ll[1]))

        f1.close()
        f2.close()
        frl.close()
        f.close()

    paper_dir = os.path.join(data_dir,"paper")
    train_path = os.path.join(paper_dir,"trn-levelone.txt.10K")
    dev_path = os.path.join(paper_dir,"dev-levelone.txt.10K")
    test_path = os.path.join(paper_dir,"tst-levelone.txt.10K")
    way4_dir = os.path.join(data_dir,"paper_4way")
    way5_dir = os.path.join(data_dir,"paper_5way")
    
    _extract(train_path, way4_dir+"/train", 4)
    _extract(dev_path, way4_dir+"/dev", 4)
    _extract(test_path, way4_dir+"/test", 4)

    _extract(train_path, way5_dir+"/train", 5)
    _extract(dev_path, way5_dir+"/dev", 5)
    _extract(test_path, way5_dir+"/test", 5)


def main():
    action = sys.argv[1]
    if action == "split":
        # generate the dev/train/test
        split()
    elif action == "statistics":
        # get the statistics of each label
        statistics()
    elif action == "human_readable":
        human_readable()
    elif action == "tokenize":
        folder = sys.argv[2]
        tokenize_data(folder)
    elif action == "extract_paper":
        # process the paper's data
        extract_paper()
        

if __name__ == "__main__":
    main()
