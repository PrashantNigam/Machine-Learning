import csv
import math

'''
    Attributes
        1. char_freq_; 
            percentage of total characters in the email that are equal to _
        2. char_freq_( 
            percentage of total characters in the email that are equal to (
        3. char_freq_[ 
            percentage of total characters in the email that are equal to [
        4. char_freq_! 
           percentage of total characters in the email that are equal to !
        5. char_freq_$ 
           percentage of total characters in the email that are equal to !
        6. char_freq_# numeric
           percentage of total characters in the email that are equal to # 
        7. capital_run_length_average 
           average length of uninterrupted sequences of capital letters
        8. capital_run_length_longest
           length of longest uninterrupted sequence of capital letters
        9. capital_run_length_total 
           total number of capital letters in the email
'''

attributes = ['char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#',
              'capital_run_length_average ', 'capital_run_length_longest', 'capital_run_length_total ']


def get_dataset(filename):
    lines = csv.reader(open(filename, 'rt'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def estimate_pc():
    dataset = get_dataset('spambasetrain.csv')
    class_0_count = 0
    class_1_count = 0
    for line in dataset:
        if line[-1] == 0:
            class_0_count += 1
        else:
            class_1_count += 1
    pc_0 = class_0_count / len(dataset)
    pc_1 = class_1_count / len(dataset)
    return pc_0, pc_1


def get_partitioned_data(filename):
    dataset = get_dataset(filename)
    xc_0 = list()
    xc_1 = list()
    for line in dataset:
        if line[-1] == 0:
            xc_0.append(line)
        else:
            xc_1.append(line)
    return xc_0, xc_1


def train():
    xc_0, xc_1 = get_partitioned_data('spambasetrain.csv')
    means = list()
    variances = list()
    # Iterate over each of the two sets
    i = 0
    for xc in [xc_0, xc_1]:
        # Iterate over columns
        line = xc[i]
        for j in range(len(line) - 1):
            # Iterate over rows to calculate mean
            mean = 0
            # row[j] is value of xi in current row
            for row in xc:
                mean += row[j]
            mean /= len(xc)

            # Iterate over rows to calculate variance
            variance = 0
            for row in xc:
                variance += math.pow((row[j] - mean), 2)
            nc = len(xc)
            if nc is not 1:
                variance /= (nc - 1)
            means.append(mean)
            variances.append(variance)
    return means, variances


def pdf(xi, mean, variance):
    exponent = -math.pow(xi - mean, 2) / (2 * variance)
    pdf_value = math.exp(exponent) / (math.sqrt(2 * math.pi * variance))
    return pdf_value


def test(means, variances, pc_0, pc_1):
    dataset = get_dataset('spambasetest.csv')
    means_class_0 = means[:len(means) // 2]
    means_class_1 = means[len(means) // 2:]
    variances_class_0 = variances[:len(variances) // 2]
    variances_class_1 = variances[len(variances) // 2:]

    count_0 = 0
    count_1 = 0
    correct_class_count = 0
    incorrect_class_count = 0
    predicted_classes = list()
    # Iterate over dataset rows
    for i in range(len(dataset)):
        # Iterate over attributes
        cum_ln_pdf_0 = math.log(pc_0)
        cum_ln_pdf_1 = math.log(pc_1)
        for j in range(len(dataset[i]) - 1):
            pdf_0 = pdf(dataset[i][j], means_class_0[j], variances_class_0[j])
            ln_pdf_0 = math.log(pdf_0)
            cum_ln_pdf_0 += ln_pdf_0

            pdf_1 = pdf(dataset[i][j], means_class_1[j], variances_class_1[j])
            ln_pdf_1 = math.log(pdf_1)
            cum_ln_pdf_1 += ln_pdf_1

        if cum_ln_pdf_0 > cum_ln_pdf_1:
            count_0 += 1
            predicted_classes.append("\tExample {0}: Class {1}".format(i + 1, 0))
            if dataset[i][len(dataset[0]) - 1] == 0:
                correct_class_count += 1
            else:
                incorrect_class_count += 1
        else:
            count_1 += 1
            predicted_classes.append("\tExample {0}: Class {1}".format(i + 1, 1))
            if (dataset[i][len(dataset[0]) - 1] == 1) or (cum_ln_pdf_0 == cum_ln_pdf_1):
                correct_class_count += 1
            else:
                incorrect_class_count += 1
    return predicted_classes, correct_class_count, incorrect_class_count


def main():
    pc_0, pc_1 = estimate_pc()
    means, variances = train()
    predicted_classes, correct_class_count, incorrect_class_count = test(means, variances, pc_0, pc_1)
    print_jobs(pc_0, pc_1, means, variances, predicted_classes, correct_class_count, incorrect_class_count)


def get_percentage_error(correct, incorrect):
    return incorrect * 100 / (correct + incorrect)


def zero_r():
    datasets = get_partitioned_data('spambasetrain.csv')
    nc_0 = len(datasets[0])
    nc_1 = len(datasets[1])
    if nc_0 > nc_1:
        accuracy = nc_0*100/(nc_0 + nc_1)
        return "Class 0", math.ceil(accuracy*100)/100
    else:
        accuracy = nc_1 * 100 / (nc_0 + nc_1)
        return "Class 1", math.ceil(accuracy*100)/100


def print_jobs(pc_0, pc_1, means, variances, predicted_classes, correct_class_count, incorrect_class_count):
    print("P(C=0) =", pc_0)
    print("P(C=1) =", pc_1)
    print("\nClass 0")
    for i in range(int(len(means) / 2)):
        print("Attribute-{0} {1}".format(i + 1, attributes[i]))
        print("\tMean =", means[i])
        print("\tVariance =", variances[i])
    print("\nClass 1")
    for i in range(int(len(means) / 2), len(means)):
        print("Attribute-{0} {1}".format(i - 8, attributes[i - 9]))
        print("\tMean =", means[i])
        print("\tVariance =", variances[i])
    print("\nPredicted Classes")
    for predicted_class in predicted_classes:
        print(predicted_class)
    print("\nTest examples classified correctly =", correct_class_count)
    print("Test examples classified incorrectly =", incorrect_class_count)
    print("Percentage error on the test examples = {0}%".format(get_percentage_error(correct_class_count,
                                                                                     incorrect_class_count)))
    label, accuracy = zero_r()
    print("By Zero R, {}\nAccuracy = {}%".format(label, accuracy))


main()
