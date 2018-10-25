import numpy as np

fpath = './remote_gold_cat.npz'

def main():
    npz = np.load(fpath)
    train = npz['train']
    print(train)

if __name__ == '__main__':
    main()