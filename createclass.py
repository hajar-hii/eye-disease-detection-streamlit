import pickle

class_indices = {
    "CATARACT": 0,
    "DIABETIC_RETINOPATHY": 1,
    "GLAUCOMA": 2,
    "NORMAL": 3
}

with open("class_indices.pkl", "wb") as f:
    pickle.dump(class_indices, f)

print("Recovered class_indices.pkl")
