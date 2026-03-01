import h5py

file_path = r"c:\Users\Asus\Desktop\training_dataset\backend\models\model.h5"
with h5py.File(file_path, 'r') as f:
    for key in f.keys():
        print(f"Layer found: {key}")
        # Let's see how many weights are in this specific layer
        try:
            weights_group = f[key][key]
            print(f"  - Weights inside: {list(weights_group.keys())}")
        except:
            pass