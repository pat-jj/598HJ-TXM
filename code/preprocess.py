from os import mkdir
import sys
import shutil
import pandas as pd


def get_keywords_wstc(source_file, out_file):
    text = ""
    with open(source_file) as f:
        lines = f.readlines()
    count = 0
    for line in lines:
        if "Category" in line:
            continue
        head = str(count) + ':'
        count += 1
        text += head
        words = line.split(" ")
        for idx in range(len(words)):
            if idx < len(words) - 2:
                ll = words[idx] + ','
                text += ll
            else:
                text += words[idx]
    text = text[:-1]
    text_file = open(out_file, 'w')
    print(text, file=text_file)


def get_classes_wstc(source_file, out_file):
    text = ""
    with open(source_file) as f:
        lines = f.readlines()
    count = 0
    for line in lines:
        head = str(count) + ':'
        count += 1
        text += head
        text += line
    text = text[:-1]
    text_file = open(out_file, 'w')
    print(text, file=text_file)


def get_dataset_csv_wstc(dataset):
    source_file = None
    if dataset == 'movies':
        source_file = "../data_process/cate_results/movies_dataset/processed_data.txt"
    elif dataset == 'news':
        source_file = "../data_process/cate_results/news_dataset/processed_data.txt"

    with open(source_file) as f:
        lines = f.readlines()
    lines_ = []
    for line in lines:
        line = line[:-1]
        lines_.append(line)
    # initialize data of lists.
    data = {'label': str(-1),
            'text': lines_}
    df = pd.DataFrame(data)

    if dataset == 'movies':
        df.to_csv(r"../data_process/data_wstc/movies/dataset.csv", header=False, index=False)
    elif dataset == 'news':
        df.to_csv(r"../data_process/data_wstc/news/dataset.csv", header=False, index=False)


def get_embedding(dataset):
    source_file = None
    if dataset == 'movies':
        source_file = "../data_process/cate_results/movies_dataset/emb_topic_w.txt"
        shutil.copy(source_file, "../data_process/data_wstc/movies/embedding.txt")
    elif dataset == 'news':
        source_file = "../data_process/cate_results/news_dataset/emb_topic_w.txt"
        shutil.copy(source_file, "../data_process/data_wstc/news/embedding.txt")


def main():
    sys.path.append("../")

    wstc_data_path = "../data_process/data_wstc"
    
    mkdir(wstc_data_path)
    mkdir(wstc_data_path + '/movies')
    mkdir(wstc_data_path + '/news')

    shutil.copy("../data_process/original_data/movies/movies_train_labels.txt",
                    "../data_process/data_wstc/movies/labels.txt")

    shutil.copy("../data_process/original_data/news/news_train_labels.txt",
                    "../data_process/data_wstc/news/labels.txt")

    get_keywords_wstc("../data_process/cate_results/movies_dataset/res_topic.txt",
                      "../data_process/data_wstc/movies/keywords.txt")

    get_keywords_wstc("../data_process/cate_results/news_dataset/res_topic.txt",
                      "../data_process/data_wstc/news/keywords.txt")

    get_classes_wstc("../data_process/original_data/movies/movies_category.txt",
                     "../data_process/data_wstc/movies/classes.txt")

    get_classes_wstc("../data_process/original_data/news/news_category.txt",
                     "../data_process/data_wstc/news/classes.txt")

    get_dataset_csv_wstc('movies')

    get_dataset_csv_wstc('news')

    get_embedding('movies')

    get_embedding('news')


if __name__ == '__main__':
    main()
