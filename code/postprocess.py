def construct_args():
    import argparse
    parser = argparse.ArgumentParser()
    # dataset selection
    parser.add_argument('--dataset', default='movies', choices=['movies', 'news'])
    # lines of results to keep
    parser.add_argument('--lines', default=100, type=int)
    # prediction result for all documents
    parser.add_argument('--pred_all', default='out.txt', type=str)
    # output file after removing results
    parser.add_argument('--out', default='test_prediction.txt', type=str)

    args = parser.parse_args()
    return args


def take_k_lines(in_file, out_file, k):
    text = ""
    with open(in_file) as f:
        lines = f.readlines()

    for i in range(k):
        text = text + lines[i]

    text_file = open(out_file, 'w', encoding='utf-8')
    print(text, file=text_file)


def main():
    args = construct_args()
    pred_all_file = '../data_process/data_wstc/' + args.dataset + '/' + args.pred_all
    output_path = '../data/' + args.dataset + '/' + args.out
    take_k_lines(pred_all_file, output_path, args.lines)


if __name__ == '__main__':
    main()