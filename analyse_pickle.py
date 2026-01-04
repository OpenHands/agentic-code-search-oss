import pickle
import json
import torch
step = 23
file_path = f'/data/user_data/adityabs/agentic_code_search/instruct_trajectorylevelexported_model/dumped_data/global_step_{step}_training_input.pkl' # Replace with the actual path to your .pkl file

def convert_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensors(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors(i) for i in obj]
    else:
        return obj

try:
    # Open the file in read-binary mode ('rb')
    with open(file_path, 'rb') as file:
        # Load the dictionary from the file
        loaded_dictionary = pickle.load(file)

    # Now you can use the loaded_dictionary
    print("Dictionary successfully loaded:")
    print(loaded_dictionary.keys())
    serializable_data = convert_tensors(loaded_dictionary)
    for k, v in serializable_data.items():
        with open(f"temp_{k}.json", "w") as f:
            json.dump(v, f)
    # with open("temp1.json", "w") as f:
    #     data = {"data": serializable_data["sequences"]}
    #     json.dump(data, f)
        # f.write()
    print(f"Type of loaded object: {type(loaded_dictionary)}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except pickle.UnpicklingError as e:
    print(f"Error unpickling the file: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

