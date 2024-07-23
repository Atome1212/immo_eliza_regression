import pandas as pd

def get_file_extension(file_path):
    return file_path.split("/")[-1].split(".")[-1]

def load_file(file_path:str):
    if get_file_extension(file_path) == "json":
        return pd.DataFrame(pd.read_json(file_path))
    
    elif get_file_extension(file_path) == "csv":
        return pd.DataFrame(pd.read_csv(file_path)) 

    else:
        return False

if __name__ == "__main__":
    print(load_file("./data/final_dataset.json"))
