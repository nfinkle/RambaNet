import csv
import numpy as np
import pathlib
import os
import youtokentome as yttm
import glob


def get_sample(tokenizer, dataset_directory = "./raw_dataset/Talmud/organized", input_size=1024, min_ratio=0.5, PAD_ID=0, authors_not_to_cut=['Talmud']):
    ds_path = pathlib.Path(dataset_directory)
    authors = list(enumerate(ds_path.iterdir()))
    for author_id, author_dir in authors:
        author = os.path.basename(author_dir)
        author_min_ratio = None if author in authors_not_to_cut else min_ratio
        print(author, author_min_ratio)
        for book_path in author_dir.iterdir():
            encoded_samples = encode_samples(tokenizer, input_size, book_path, author_min_ratio, PAD_ID)
            for sample in encoded_samples:
                yield sample, author_id


def split_and_pad_array(list, size, min_size, PAD_ID=0):
    split_arr = np.array_split(np.array(list), range(size, len(list), size))
    last_arr = split_arr[-1]
    if len(last_arr) < min_size:
        split_arr = split_arr[:-1]
        if len(split_arr) == 0:
            return np.empty(0, dtype=int)
        # print(len(list), size, min_size, [s.shape for s in split_arr])
        return np.stack(split_arr)
    if len(last_arr) < size:
        split_arr[-1] = np.pad(last_arr, (0, size - len(last_arr)), 'constant',constant_values=PAD_ID)
    return np.stack(split_arr, axis=0)


def encode_samples(tokenizer, input_size, book_path, min_ratio, PAD_ID=0):
    with book_path.open(mode ='r', encoding='utf8') as book_file:
        lines = book_file.read().splitlines()

    min_length = 0 if min_ratio is None else int(input_size * min_ratio) 
    lines = list(filter(lambda x: x.strip(), lines))
    lines = tokenizer.encode(lines, output_type=yttm.OutputType.ID, bos=True, eos=True) # list of int lists
    padded_ids = list(filter(lambda x: len(x) > 0, [split_and_pad_array(line, input_size, min_length, PAD_ID) for line in lines])) # list of list of np.arrays of differing shapes
    padded_ids = [' '.join(map(str, s)) for s in np.concatenate(padded_ids, axis=0)]
    return padded_ids


def load_tokenizer(model_path):
  return yttm.BPE(model=model_path)


def create_tokenizer(data_path, output_path, vocab_size):
  return yttm.BPE.train(data=data_path, model=output_path, vocab_size=vocab_size)


def write_dataset_to_tsv(output_file_path, generator_fn):
    # Open output file for writing
    with open(output_file_path, mode="w", newline="", encoding="utf-8") as output_file:
    # Create TSV writer
        tsv_writer = csv.writer(output_file, delimiter="\t")

    # Write header row (optional)
        # tsv_writer.writerow(["Encoded Example", "Label"])

    # Iterate over generator function and write to TSV file
        for input_text, target_text in generator_fn():
            tsv_writer.writerow([input_text, target_text])


# Open the output file for writing
def combine_all_files(root_dir, output_file):
    with open(output_file, 'w') as output_file:
    # Iterate over all text files in the directory tree
        for file_path in glob.iglob(os.path.join(root_dir, '**/*.txt'), recursive=True):
        # Open the text file for reading
            with open(file_path, 'r') as input_file:
            # Append the contents of the text file to the output file
                output_file.write(input_file.read())
