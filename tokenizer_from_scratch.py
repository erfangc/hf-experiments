from datasets import load_dataset

raw_datasets = load_dataset('code_search_net', 'python')


def training_data():
    batch_size = 3
    for i in range(0, len(raw_datasets["train"]), batch_size):
        yield raw_datasets["train"][i:i + batch_size - 1]["whole_func_string"]


print("Getting item ...")
print(next(training_data()))
print(next(training_data()))
