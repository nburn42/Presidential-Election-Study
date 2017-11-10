import random


class dataset:
    def __init__(self):
        header = None
        linear_header = []
        labels_header = []

        linear_data = []
        data = []
        raw_labels = []
        labels = []

        last_labels = []

        mins = []
        maxs = []

        linear_mins = []
        linear_maxs = []

        linear_columns = [-1, 10, 7, 6, 4, 0]

        with open("smalldataset.csv", "r") as f:
            for line in f:
                if header is None:
                    header = line.strip().split(",")
                    labels_header = header[-12:]
                    header = header[:-12]
                    header.pop(1)

                    for i in linear_columns:
                        linear_header.append(header.pop(i))

                    mins = [99 ** 99 for x in range(len(header))]
                    maxs = [0 for x in range(len(header))]
                    linear_mins = [99 ** 99 for x in range(len(header))]
                    linear_maxs = [0 for x in range(len(header))]
                    continue
                d = list(map(int, line.strip().split(",")))
                dd = d[:-12]

                dd.pop(1)

                ld = []
                for i in linear_columns:
                    ld.append(dd.pop(i))

                data.append(dd)
                linear_mins = [float(min(x, y)) for x, y in zip(linear_mins, ld)]
                linear_maxs = [float(max(x, y)) for x, y in zip(linear_maxs, ld)]
                linear_data.append(ld)
                mins = [min(x, y) for x, y in zip(mins, dd)]
                maxs = [max(x, y) for x, y in zip(maxs, dd)]
                raw_labels.append(d[-12:])

        new_linear_data = []
        for line in linear_data:
            dd = []
            for i, l in enumerate(line):
                dd.append((l - linear_mins[i]) / (0.0002 + linear_maxs[i] - linear_mins[i]))
            new_linear_data.append(dd)

        linear_data = new_linear_data

        print(list(zip(header, data[0])))
        print(list(zip(header, mins)))
        print(list(zip(header, maxs)))
        print(list(zip(linear_header, linear_data[0])))
        print(list(zip(linear_header, linear_mins)))
        print(list(zip(linear_header, linear_maxs)))
        print(list(zip(labels_header, raw_labels[-50])))

        for ind, d in enumerate(data):
            for i in range(len(d)):
                dd = [0 for x in range(1 + maxs[i] - mins[i])]
                dd[d[i] - mins[i]] = 1
                linear_data[ind].extend(dd)

        useful_indexes = [[] for x in labels_header]

        for ind, l in enumerate(raw_labels):
            labels.append([])
            last_labels.append([])
            for i in range(len(l)):
                dd = [0 for x in range(2)]
                if (l[i] == 1 or l[i] == 2):
                    dd[l[i] - 1] = 1
                    useful_indexes[i].append(ind)
                labels[ind].extend(dd)
                if i == len(l) - 1:
                    last_labels[ind].extend(dd)

        print(map(len, useful_indexes))

        final_data = []
        final_labels = []

        for ind in useful_indexes[-1]:
            final_data.append(linear_data[ind])
            final_labels.append(last_labels[ind])

        dataset = list(zip(final_data, final_labels))
        random.shuffle(dataset)
        test_length = int(len(dataset) * 0.67)

        print("test_length", test_length)
        self.train_dataset = dataset[:test_length]
        self.test_dataset = dataset[test_length:]
