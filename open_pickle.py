import pickle

with open('data/BF-PSR-Framework-Data/PJZ.pkl', 'rb') as f:
    pjz_data = pickle.load(f)

print(pjz_data)