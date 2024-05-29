import csv
import os.path

def csv_writer(csv_file, folder_save):
    file_negative = './data/' + folder_save
    file_neutral = './data/' + folder_save
    file_positive = './data/' + folder_save

    if not os.path.isdir(file_negative):
        os.makedirs(file_negative)
    if not os.path.isdir(file_neutral):
        os.makedirs(file_neutral)
    if not os.path.isdir(file_positive):
        os.makedirs(file_positive)

    file_negative = file_negative + '/rt-polarity.neg'
    file_neutral = file_neutral + '/rt-polarity.neu'
    file_positive = file_positive + '/rt-polarity.pos'

    string_negative = ''
    string_neutral = ''
    string_positive = ''

    with open(csv_file) as f:
        reader = csv.reader(f, delimiter='\t')
        index = 2
        fields = len(next(reader))

        if fields != 4:
            index = 1

        for row in reader:
            if row[index] == 'neutral':
                if row[index + 1] != 'Not Available':
                    string_neutral += row[index + 1] + '.\r\n'
            elif row[index] == 'negative':
                if row[index + 1] != 'Not Available':
                    string_negative += row[index + 1] + '.\r\n'
            else:
                if row[index + 1] != 'Not Available':
                    string_positive += row[index + 1] + '.\r\n'

    text_file = open(file_negative, 'w')
    text_file.write(string_negative)
    text_file.close()

    text_file = open(file_neutral, 'w')
    text_file.write(string_neutral)
    text_file.close()

    text_file = open(file_positive, 'w')
    text_file.write(string_positive)
    text_file.close()


def MergeFile(filenames, writefile):

    if not os.path.isdir('./data/data_3/train'):
        os.makedirs('./data/data_3/train')
    if not os.path.isdir('./data/data_3/test'):
        os.makedirs('./data/data_3/test')

    with open(writefile, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


if __name__ == "__main__":
    # Data Preparation
    print("Preparation data...")

    # # data_1
    # # train
    # csv_writer('./data_raw/data_1/twitter-train-cleansed-B.tsv', 'data_1/train')
    # # test
    # csv_writer('./data_raw/data_1/twitter-test-gold-B.tsv', 'data_1/test')
    #
    # # small data
    # # train
    # csv_writer('./data_raw/data_1/twitter-train-small.tsv', 'data_1/train-small')
    # # test
    # csv_writer('./data_raw/data_1/twitter-test-small.tsv', 'data_1/test-small')
    # print('Files preparation data from csv file for data_1 successfully written.')
    #
    # # data_2
    # # train
    # csv_writer('./data_raw/data_2/100_topics_100_tweets.sentence-three-point.subtask-A.train.gold.tsv', 'data_2/train')
    # # test
    # csv_writer('./data_raw/data_2/100_topics_100_tweets.sentence-three-point.subtask-A.test.gold.tsv', 'data_2/test')
    # print('Files preparation data from csv file for data_2 successfully written.')

    filenames = ['./data/data_1/train/rt-polarity.neg', './data/data_2/train/rt-polarity.neg']
    MergeFile(filenames, './data/data_3/train/rt-polarity.neg')

    filenames = ['./data/data_1/train/rt-polarity.neu', './data/data_2/train/rt-polarity.neu']
    MergeFile(filenames, './data/data_3/train/rt-polarity.neu')

    filenames = ['./data/data_1/train/rt-polarity.pos', './data/data_2/train/rt-polarity.pos']
    MergeFile(filenames, './data/data_3/train/rt-polarity.pos')

    filenames = ['./data/data_1/test/rt-polarity.neg', './data/data_2/test/rt-polarity.neg']
    MergeFile(filenames, './data/data_3/test/rt-polarity.neg')

    filenames = ['./data/data_1/test/rt-polarity.neu', './data/data_2/test/rt-polarity.neu']
    MergeFile(filenames, './data/data_3/test/rt-polarity.neu')

    filenames = ['./data/data_1/test/rt-polarity.pos', './data/data_2/test/rt-polarity.pos']
    MergeFile(filenames, './data/data_3/test/rt-polarity.pos')



