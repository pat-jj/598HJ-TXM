def evaluate(true_path, pred_path, lines):
    with open(true_path) as f:
        true_data = f.readlines()
    with open(pred_path) as f:
        pred_data = f.readlines()
    correct_count = 0
    for i in range(lines):
        if true_data[i] == pred_data[i]:
            correct_count += 1
    acc = correct_count / lines
    eval_res = "Accuracy of the prediction: " + acc
    return eval_res


def construct_args():
    import argparse
    parser = argparse.ArgumentParser()
    # dataset selection
    parser.add_argument('--dataset', default='movies', choices=['movies', 'news'])
    # lines of results to evaluate
    parser.add_argument('--lines', default=100, type=int)
    # name of the text file containing true labels
    parser.add_argument('--labels', default='labels.txt', type=str)
    # name of the text file containing inference results
    parser.add_argument('--pred', default='out.txt', type=str)

    args = parser.parse_args()
    return args


def main():
    args = construct_args()
    current = '../data_process/data_wstc/'
    true_file = current + args.dataset + '/' + args.labels
    pred_file = current + args.dataset + '/' + args.pred
    print(evaluate(true_file, pred_file, args.lines))


if __name__ == '__main__':
    main()
