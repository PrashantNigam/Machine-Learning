import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity


def get_token_sets(filename):
    with open(filename, 'rt') as file:
        labels = list()
        token_sets = list()
        for line in file:
            labels.append(int(line[0]))
            token_set = set(line[2:].split())
            token_sets.append(token_set)
    return labels, token_sets


def get_lines(filename):
    with open(filename, 'rt') as file:
        labels = list()
        lines = list()
        for line in file:
            labels.append(int(line[0]))
            lines.append(line[2:].strip())
    return labels, lines


def k_nearest_neighbours(k, train_token_sets, train_labels, test_token_set):
    k_nearest = list()
    for i in range(len(train_labels)):
        common_tokens = train_token_sets[i].intersection(test_token_set)
        if len(common_tokens):
            k_nearest.append((train_labels[i], common_tokens))
    k_nearest.sort(key=lambda item: -len(item[1]))

    if len(k_nearest) >= k:
        farthest_distance = len(k_nearest[k - 1][1])
        i = k - 1
        while i < len(k_nearest):
            if len(k_nearest[i][1]) == farthest_distance:
                i += 1
            else:
                break
        k_nearest = k_nearest[0:i]

    label_zero = 0
    label_one = 0
    for x in k_nearest:
        if x[0] == 0:
            label_zero += 1
        else:
            label_one += 1

    if len(k_nearest):
        predicted_label = 0 if label_zero > label_one else 1
        return predicted_label
    else:
        return None


def one_a():
    test_file = 'reviewstest.txt'
    actual_labels, test_token_sets = get_token_sets(test_file)

    train_file = 'reviewstrain.txt'
    train_labels, train_token_sets = get_token_sets(train_file)

    for k in [1, 5]:
        print('For k = {}:'.format(k))
        i = 0
        predicted_labels = list()
        for test_token_set in test_token_sets:
            predicted_label = k_nearest_neighbours(k, train_token_sets, train_labels, test_token_set)
            predicted_labels.append(predicted_label)
            if i == 17:
                print('Predicted Label: ', predicted_label)
            i += 1
        find_measures(confusion_matrix(actual_labels, predicted_labels))


def find_measures(cm):
    print(cm)
    print('True Positive = ', cm[1][1])
    print('False Positive = ', cm[0][1])
    accuracy = ((cm[1][1] + cm[0][0]) * 100 / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]))
    print('Accuracy = {}%\n'.format(accuracy))
    return accuracy


def zero_r():
    actual_labels = get_token_sets('reviewstest.txt')[0]
    label_zero, label_one = 0, 0
    with open('reviewstrain.txt', 'rt') as file:
        for line in file:
            if line[0] == 0:
                label_zero += 1
            else:
                label_one += 1
    dominant_label = 0 if label_zero > label_one else 1
    predicted_labels = [dominant_label] * len(actual_labels)
    print('\nZero-R\n', confusion_matrix(actual_labels, predicted_labels))


def cross_validation():
    train_labels, train_token_sets = get_token_sets('reviewstrain.txt')
    label_chunks = get_chunks(train_labels, 5)
    token_set_chunks = get_chunks(train_token_sets, 5)

    max_accuracy = 0
    for k in [3, 7, 99]:
        predicted_labels = list()
        for i in range(5):
            for test_token_set in token_set_chunks[i]:
                combined_test_token_chunks = list()
                combined_test_label_chunks = list()
                for j in range(5):
                    if i != j:
                        combined_test_token_chunks.extend(token_set_chunks[j])
                        combined_test_label_chunks.extend(label_chunks[j])
                predicted_label = k_nearest_neighbours(k, combined_test_token_chunks, combined_test_label_chunks,
                                                       test_token_set)
                predicted_labels.append(predicted_label)
        accuracy = find_measures(confusion_matrix(train_labels, predicted_labels))
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_k = k

    actual_labels, test_token_sets = get_token_sets('reviewstest.txt')
    predicted_labels = list()
    for test_token_set in test_token_sets:
        predicted_label = k_nearest_neighbours(best_k, train_token_sets, train_labels, test_token_set)
        predicted_labels.append(predicted_label)
    find_measures(confusion_matrix(actual_labels, predicted_labels))


def get_chunks(li, n):
    chunks = [np.array(x).tolist() for x in np.array_split(np.array(li), n)]
    return chunks


def get_cosine_similarity():
    actual_labels, test_lines = get_lines('reviewstest.txt')
    train_labels, train_lines = get_lines('reviewstrain.txt')

    vectorizer = TfidfVectorizer()

    for k in [1, 5]:
        print('\n k = ', k)
        i = 0
        predicted_labels = list()
        for test_line in test_lines:
            temp = train_lines.copy()
            temp.insert(0, test_line)
            train_matrix = vectorizer.fit_transform(temp)

            k_nearest = list()
            similarities = cosine_similarity(train_matrix[0], train_matrix)[0]
            for similarity in similarities:
                k_nearest.append((train_labels[i], similarity))
            k_nearest.sort(key=lambda x: -x[1])
            k_nearest.pop(0)
            i += 1

            farthest = k_nearest[k - 1]
            j = k
            while j < len(k_nearest):
                if k_nearest[j] == farthest:
                    j += 1
                else:
                    break
            k_nearest = k_nearest[:j]

            label_zero, label_one = 0, 0
            for item in k_nearest:
                if item[0] == 0:
                    label_zero += 1
                else:
                    label_one += 1
            predicted_label = 0 if label_zero > label_one else 1
            predicted_labels.append(predicted_label)
        find_measures(confusion_matrix(actual_labels, predicted_labels))


if __name__ == '__main__':
    # one_a()
    # zero_r()
    # cross_validation()

    prompt = 'Press \n\t1 for default distance function\n\t2 for cosine-similarity distance function\n'
    while True:
        try:
            choice = int(input(prompt))
            if choice != 1 and choice != 2:
                raise ValueError
            break
        except ValueError:
            print('Invalid Option! Please Try Again.')
            continue

    if choice == 1:
        one_a()
    else:
        get_cosine_similarity()
