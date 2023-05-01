import json
import pathlib
import shutil
import re
import numpy as np
import tables
import tensorflow as tf
import os
from utils import flatten, str2onehot, text_example, split_and_pad_strings
import io


#Organize dataset
#TODO: consider write to TFRecord file
# https://towardsdatascience.com/working-with-tfrecords-and-tf-train-example-36d111b3ff4d
# https://medium.com/@prasad.pai/how-to-use-tfrecord-with-datasets-and-iterators-in-tensorflow-with-code-samples-ffee57d298af
# https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
# https://www.tensorflow.org/tutorials/load_data/tfrecord
# https://www.tensorflow.org/guide/data
# https://www.tensorflow.org/guide/data_performance
# https://www.tensorflow.org/tutorials/load_data/tfrecord
def organize_data(dataset_dirname = "./sample_dataset/", alphabet = 'אבגדהוזחטיכךלמםנןסעפףצץקרשת \'\n"'):
    """ 
    Organize dataset in convenient folder structure and keep only relevant files in convenient form. 
    Existing data is overwritten.
    """
    #TODO: explain folder structure of input and output
    # dataset_dirname = "./sample_dataset/"
    raw_subdirname = "raw/"
    raw_metadata_subdirame = "_schemas/"
    organized_subdirname = "organized/"

    #load all relevant metadata
    authors_dict = {}
    error_count=0
    metadata_dir = dataset_dirname + raw_subdirname + raw_metadata_subdirame
    for metadata_fn in os.listdir(metadata_dir):
        filename, file_extension = os.path.splitext(metadata_fn)
        if file_extension != '.json':
            print("Skipping non-json", filename)
            continue
        with open(metadata_dir+metadata_fn, 'r', encoding="utf8") as metadata_file:
            try:
                metadata = json.load(metadata_file)
                # if 'authors' not in metadata or not metadata.get('authors'):
                #     print(metadata.get('title'), metadata.keys(), metadata.get('authors'))
            except:
                print("Could not load metadata", metadata_fn)
                continue
        bookname = filename.replace('_', ' ')
        # author = ""
        try:
            author = metadata['authors'][0]['en']
        except: #if author=='':
            idx = bookname.find(" on ")
            if idx>0:
                author = bookname[0:idx]
            else:
                author = bookname
            # print("Book '" + filename +"' has no valid author information")
            error_count+=1
        authors_dict[bookname] = author
    # print(authors_dict)
    print(str(error_count) + ' books out of '+ str(len(os.listdir(metadata_dir))) +' without valid author information were corrected. ')

    #get list of all directories in raw folder
    books = os.listdir(dataset_dirname + raw_subdirname)
    #remove metadata directory from list
    books.remove(raw_metadata_subdirame.replace('/', ''))
    books.remove('_links')
    vowels_pattern = re.compile(r'[\u05B0-\u05BD\u05BF\u05C1\u05C2\u05C7]')
    quotation_marks = re.compile("[“”״]")
    apostrophe_marks = re.compile("[‘’׳]")
    #organize books
    for book in books:
        book_path = pathlib.Path(dataset_dirname + raw_subdirname+book+'/Hebrew/merged.json')
        #validate
        if not book_path.is_file():
            print('Directory ' + str(book_path) + ' ignored (invalid location).')
            continue
        if book_path.suffix != '.json':
            print('File ' + str(book_path) + ' ignored (type should be JSON).')
            continue
        #get author
        if book in authors_dict.keys():
            curr_author = authors_dict[book]
            author_dirname = dataset_dirname+organized_subdirname+curr_author+'/'
            pathlib.Path(author_dirname).mkdir(parents=True, exist_ok=True)
            out_file = author_dirname+book+".txt"
            # shutil.copyfile(dataset_dirname + raw_subdirname+book+'/Hebrew/merged.json',)

            # Put content into simple TXT file
            # Load JSON data
            with book_path.open(mode='r', encoding='utf8') as book_file:
                try:
                    book_raw_text = json.load(book_file)['text']
                except:
                    print('File '+str(book_path) +' ignored (impossible to read JSON).')
                    continue

            # flatten
            if isinstance(book_raw_text, list) or isinstance(book_raw_text, dict):  # no internal separation of text
                flattened_raw_lst = flatten(book_raw_text)
            # elif isinstance(book_raw_text, dict):# internal separation of text - dict of dicts
            #     tmp = []
            #     for d in book_raw_text.values():
            #         if isinstance(d, dict):
            #             tmp.extend(list(d.values()))
            #         elif isinstance(d, list):
            #             tmp.extend(d)
            #     flattened_raw_lst = list(flatten(tmp))
            else:
                raise ValueError(str(book_path)+ ': Could not parse.')
            
            # ensure file does not have different structure from expected
            
            # print(get_nested_type(flattened_raw_lst))
            assert all(isinstance(x, str) for x in flattened_raw_lst)#, get_nested_type(flattened_raw_lst)
            # TODO: check manually all is well

            # concatenate
            flattened_raw_str = '\n'.join(flattened_raw_lst)

            flattened_raw_str = re.sub(vowels_pattern, '', flattened_raw_str)
            flattened_raw_str = re.sub(quotation_marks, '"', flattened_raw_str)
            flattened_raw_str = re.sub(apostrophe_marks, "'", flattened_raw_str)


            new_str = re.sub('[\t ]+', ' ', flattened_raw_str)
            new_str = re.sub('<[^<]+?>', '', new_str)
            new_str = re.sub(r'\([^)]*\)|\[[^]]*\]', '', new_str)
            # new_str = re.sub(r'[-—\?\.,׃:;!*#&a-zA-Z0-9…$%¨]', '', new_str)
            new_str = re.sub(r'[\u05D0-\u05EA\u05F0-\u05F4]\)|\(', '', new_str) #
            new_str = re.sub('[^'+alphabet+']', '', new_str)

            if len(new_str) < 10:
                print("Could not remove HTML in", out_file)
            else: 
                flattened_raw_str = new_str
            
            #write to file
            with io.open(out_file, 'w', encoding='utf8') as f:
                f.write(flattened_raw_str)

#generator function (including preprocessing -> NumPy arrays)
#TODO: consider making preprocessing after generation. For now most compatible
def get_sample(dataset_directory = "./raw_dataset/Rishonim/organized", input_size=1024, alphabet='אבגדהוזחטיכךלמםנןסעפףצץקרשת \'"'):
    ds_path = pathlib.Path(dataset_directory)
    authors = list(enumerate(ds_path.iterdir()))
    one_hot_matrix = np.eye(len(authors), dtype='int8')
    for author_id, author_dir in authors:
        print(author_dir)
        for book_path in author_dir.iterdir():
            samples_onehot = make_samples(input_size, alphabet, book_path)
            author_label = one_hot_matrix[:,author_id]
            for sample in samples_onehot:
                yield (sample, author_label)

def make_samples(input_size, alphabet, book_path):
    with book_path.open(mode    ='r', encoding='utf8') as book_file:
        flattened_raw_str = book_file.read()

    #split by paragraph
    samples = split_and_pad_strings(flattened_raw_str.split("\n"), input_size)

    #convert to numerical one-hot
    #TODO: consider convert to sparse representation
    samples_onehot = np.stack([str2onehot(sample, alphabet, input_size) for sample in samples], axis=0)
    return samples_onehot


def parse_dataset(dataset_directory = "./raw_dataset/Rishonim/organized", input_size=1024, alphabet='אבגדהוזחטיכךלמםנןסעפףצץקרשת \'"'):
    ds_path = pathlib.Path(dataset_directory)
    authors = list(enumerate(ds_path.iterdir()))
    one_hot_matrix = np.eye(len(authors), dtype='int8')
    examples = np.zeros((1, input_size, len(alphabet)))
    labels = np.zeros((1, len(authors)))

    for author_id, author_dir in authors:
        print(author_dir)
        for book_path in author_dir.iterdir():
            samples_onehot = make_samples(input_size, alphabet, book_path)
            author_label = one_hot_matrix[:,author_id]
            examples = np.concatenate((examples, samples_onehot))
            author_label = np.repeat(author_label[None, :], len(samples_onehot), axis=0)
            labels = np.concatenate((labels, author_label))

    return examples, labels


def preprocess_all_data(dataset_directory, input_size=1024, alphabet='אבגדהוזחטיכךלמםנןסעפףצץקרשת \'"', output_filename='./sample_dataset/sample_dataset'):
    """ 
    Gets dataset directory path (which has structure as detailed behind TODO), and writes to file data as numeric NumPy ndarray in HDF5 file and TFrecord file.

    If the output files already exists the preprocessed data is *overwritten*.
    """
    #initialize variables
    preprocessed_samples = np.array([], dtype=np.int8)
    preprocessed_labels =  np.array([], dtype=np.int8)

    h5_fn = output_filename+'.h5'
    tfr_fn = output_filename+'.tfrecords'

    #initialize files dataset will be stored in
    with tables.open_file(h5_fn, mode='w') as h5file, tf.io.TFRecordWriter(tfr_fn) as tfwriter:
        typeAtom = tables.Int8Atom()
        print('Processing...')
        #iterate over authors
        ds_path = pathlib.Path(dataset_directory)
        for author_label, author_dir in enumerate(ds_path.iterdir()):
            #validate
            print('Processing ' + str(author_dir) + '...')
            if not author_dir.is_dir():
                print('File '+str(author_dir) +' ignored (invalid location).')
                continue

            #create h5 group and table
            gauthor = h5file.create_group(h5file.root, 'author'+str(author_label), author_dir.name)
            array_c = h5file.create_earray(gauthor, 'samples', typeAtom, (0,len(alphabet), input_size), author_dir.name+" Samples")

            # author_dict[author_label] = author_dir.name
            for book_path in author_dir.iterdir():
                # validation check
                if not book_path.is_file():
                    print('Directory ' + str(author_dir) + ' ignored (invalid location).')
                    continue
                if book_path.suffix != '.json':
                    print('File ' + str(author_dir) + ' ignored (type should be JSON).')
                    continue

                # load JSON data
                with book_path.open(mode='r', encoding='utf8') as book_file:
                    try:
                        book_raw_text = json.load(book_file)['text']
                        # book_raw_text = book_raw_data
                    except:
                        print('File '+str(author_dir) +
                                    ' ignored (impossible to read JSON).')
                        continue

                # flatten
                if isinstance(book_raw_text, list):  # no internal separation of text
                    flattened_raw_lst = list(flatten(book_raw_text))
                elif isinstance(book_raw_text, dict):# internal separation of text - dict of dicts
                    tmp = []
                    for d in book_raw_text.values():
                        if isinstance(d, dict):
                            tmp.extend(list(d.values()))
                        elif isinstance(d, list):
                            tmp.extend(d)
                    flattened_raw_lst = list(flatten(tmp))
                else:
                    raise ValueError(str(book_path)+ ': Could not parse.')

                # ensure file does not have different structure from expected
                assert(all(isinstance(x, str) for x in flattened_raw_lst))
                # TODO: check manually all is well

                # concatenate
                flattened_raw_str = ''.join(flattened_raw_lst)

                # TODO: handle single quote characters

                # keep only letters in alphabet and remove multiple spaces
                filtered = re.sub('[^'+alphabet+']', ' ', flattened_raw_str)
                # TODO: is it always correct to replace out-of-alphabet characters by spaces?

                # split to samples
                #TODO: prevent cutting in the middle of words
                n = input_size
                samples = [filtered[i:i+n] for i in range(0, len(filtered), n)]

                #convert to numerical one-hot
                samples_onehot_minus1 = np.stack([str2onehot(sample, alphabet, input_size) for sample in samples[0:-1]], axis=0)
                #pad last sample and add it to 3d array
                lastsample_onehot = str2onehot(samples[-1], alphabet, input_size)
                lastsample_onehot_padded = np.zeros_like(samples_onehot_minus1[-1,:,:], dtype=np.int8)
                lastsample_onehot_padded[0:lastsample_onehot.shape[0], 0:lastsample_onehot.shape[1]] = lastsample_onehot
                samples_onehot = np.concatenate((samples_onehot_minus1, lastsample_onehot_padded[np.newaxis,:,:]))

                ## write to file
                #write to h5
                array_c.append(samples_onehot)
                #write to tfrecord
                for text_arr in samples_onehot:
                    tf_example = text_example(text_arr, author_label)
                    tfwriter.write(tf_example.SerializeToString())
            h5file.flush()
            tfwriter.flush()


def move_books_to_main(main_dir="/Users/nathan/Library/CloudStorage/Dropbox/_School/College/10_-_Masters_Spring/COS485/Final Project/RambaNet/raw_dataset"):
    import shutil
    def traverse_cur_dir(cur_dir, prefix):
        if not os.path.isdir(prefix + cur_dir):
            print(prefix + cur_dir, "not a directory")
            return
        if "Hebrew" in os.listdir(prefix + cur_dir):

            shutil.move(prefix+cur_dir, main_dir+ "/" + cur_dir.split("/")[-1])
        else:
            for f in os.listdir(prefix + cur_dir):
                traverse_cur_dir(f, prefix + cur_dir +"/")
    
    names  = main_dir.split("/")
    traverse_cur_dir(names[-1], "/".join(names[:-1]) + "/")

def linebreaks_at_colons_and_breakup_html(organized_folder):
    import glob
    import re
    pattern = re.compile(r'[\u05B0-\u05BD\u05BF\u05C1\u05C2]')
    for filename in glob.glob(organized_folder + "/*/*.txt"):
        filedata = None
        with open(filename, 'r') as file:
            filedata = file.read()

        # Replace the target string
        filedata = re.sub(r"<.*>", "", filedata)
        filedata = re.sub(r"\(.*\)", "", filedata)
    # Remove all matches from the string and return the result
        filedata = re.sub(pattern, '', filedata)

        filedata = filedata.split(":")

        # Write the file out again
        with open(filename, 'w') as file:
            file.write('\n'.join(filedata))


def remove_incorrect_authorship(starting_dir):
    REMOVE = ['Rashi on Taanit.txt', 'Rashi on Nedarim.txt', 'Rashi on Nazir.txt', 'Rashi on Horayot.txt', "Nuschaot Ktav Yad"]
    for name in REMOVE:
        for root, dirs, files in os.walk(starting_dir):
            for file in files:
                if file.endswith(name):
                    p = os.path.join(root, file)
                    print("removing", p)
                    os.remove(p)
            for dir in dirs:
                if dir.endswith(name):
                    p = os.path.join(root, dir)
                    print("removing", p)
                    shutil.rmtree(p)

def move_maschetot_to_talmud(starting_dir):
    # List of strings to match with directory names
    match_list = [ 'Berakhot', 'Shabbat', 'Eruvin', 'Pesachim', 'Shekalim', 'Yoma', 'Sukkah', 'Beitzah', 'Rosh Hashanah', 'Taanit', 'Megillah', 'Moed Katan', 'Chagigah', 'Yevamot', 'Ketubot', 'Nedarim', 'Nazir', 'Sotah', 'Gittin', 'Kiddushin', 'Bava Kamma', 'Bava Metzia', 'Bava Batra', 'Sanhedrin', 'Makkot', 'Shevuot', 'Avodah Zarah', 'Horayot', 'Zevachim', 'Menachot', 'Chullin', 'Bekhorot', 'Arakhin', 'Temurah', 'Keritot', 'Meilah', 'Niddah', 'Tamid', 'Negaim', 'Niddah']

    # Destination directory where matched files will be moved
    destination_dir = "Talmud"

    move_files_from_dir(starting_dir, match_list, destination_dir)


def move_files_from_dir(starting_dir, match_list, destination_dir):
    dest_dir = os.path.join(starting_dir, destination_dir)

    # Loop over subdirectories in the source directory
    for subdir in os.listdir(starting_dir):
        # Check if subdirectory name matches with any string in the match list
        if any(x in subdir for x in match_list):
            # Full path of the subdirectory
            subdir_path = os.path.join(starting_dir, subdir)
            # Loop over files in the subdirectory
            for file in os.listdir(subdir_path):
                # Full path of the file
                file_path = os.path.join(subdir_path, file)
                # Check if it is a file and not a subdirectory
                if os.path.isfile(file_path):
                    # Move the file to the destination directory
                    shutil.move(file_path, dest_dir)
            os.rmdir(subdir_path)


def delete_DS_STORE(starting_dir):
    for root, dirs, files in os.walk(starting_dir):
        for file in files:
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))


def do_it_all(starting_dir="./raw_dataset/Talmud/"):
    organized_path = os.path.join(starting_dir, "organized")
    shutil.rmtree(organized_path)
    delete_DS_STORE(starting_dir)
    organize_data(starting_dir)
    inside_talmud_dir = os.path.join(organized_path, "Talmud")
    if os.path.exists(inside_talmud_dir):
        shutil.rmtree(inside_talmud_dir)
    os.mkdir(inside_talmud_dir)
    remove_incorrect_authorship(organized_path)
    move_maschetot_to_talmud(organized_path)
    move_files_from_dir(organized_path, ["Hilkhot HaRamban"], "Ramban")