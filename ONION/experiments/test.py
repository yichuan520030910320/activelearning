import os
def myjoin(a,b):
    return a+'/'+b
os.path.join = myjoin
def get_all_data(base_path):

    train_path = os.path.join(base_path, 'train.tsv')
    return train_path
if __name__== '__main__':
    print(get_all_data("./data/clean_data/sst-2"))
