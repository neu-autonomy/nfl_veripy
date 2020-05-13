import pickle

def save_dataset(xs, us):
    with open("dataset.pkl", "wb") as f:
        pickle.dump([xs, us], f)

def load_dataset():
    with open("/Users/mfe/Downloads/dataset.pkl", "rb") as f:
        xs, us = pickle.load(f)
    return xs, us

if __name__ == '__main__':
    save_dataset(xs, us)
    xs, us = load_dataset()