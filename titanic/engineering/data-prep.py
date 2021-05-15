import csv
import math

filename = "train"
training = "train" == filename
onehot_encode = False
filepath = "data/original/{}.csv".format(filename)
outputpath = "data/prepped/{}-{}.csv".format(filename, 'ohe' if onehot_encode else 'basic')
column_offset = 1 if training else 0


def prepare_record(row: [str]) -> [float]:
    pclass = row[0]
    name = row[1]
    sex = row[2]
    age_str = row[3]
    sibsp = row[4]
    parch = row[5]
    ticket = row[6]
    fare = row[7]
    cabin = row[8]
    embarked = row[9]
    male = 1 if sex == 'male' else 0
    parents_or_children = int(parch)
    siblings_or_spouses = int(sibsp)
    if fare == "":
        fare = "0"
    fare_paid = float(fare)
    if age_str == "":
        if "Master" in name or "master" in name:
            age_str = '2'
        elif "Mrs." in name:
            age_str = '30'
        elif parents_or_children < 1:
            age_str = "30"
        elif parents_or_children < 2:
            if siblings_or_spouses > 1:
                age_str = "7"
            else:
                age_str = "29"
        elif parents_or_children < 3:
            if siblings_or_spouses > 2:
                age_str = "10"
            else:
                age_str = "20"
        else:
            age_str = "30"
    age = int(math.floor(float(age_str)))

    # Now derived fields
    embark_num = 0
    if embarked == 'S':
        embark_num = 1
    if embarked == 'Q':
        embark_num = 2
    fare_dollar = int(round(fare_paid))
    their_class = int(pclass)
    return [male, age, fare_dollar, embark_num, their_class, parents_or_children]


with open(outputpath, "w") as output:
    with open(filepath) as input:
        reader = csv.reader(input)
        first = True
        anything_written = False
        for row in reader:
            if first:
                first = False
                continue
            row_id = row[0]
            if training:
                survived = row[1]
                in_features = row[2:]
            else:
                in_features = row[1:]
            record = prepare_record(in_features)
            if anything_written:
                output.write("\n")
            if training:
                output.write(str(survived) + ",")
            data = [str(x) for x in record]
            line = ",".join(data)
            output.write(line)
            if not training:
                output.write("," + str(row_id))
            anything_written = True

# Ideas: find features that are highly correlated, see if you can combine them.  See if you can reduce dimensions.
