import csv
import numpy as np
import sys


if __name__ == '__main__':
    infile = sys.argv[1]
    infile_testing = sys.argv[2]
    tribute_index= sys.argv[3]
    train_out= sys.argv[4]
    test_out = sys.argv[5]
    metrics_out= sys.argv[6]

tribute_index=int(tribute_index)

def openFile(infile,tribute_index):
    with open(infile) as f1:
        reader = csv.reader(f1, delimiter='\t')
        data=[]
        for row in reader:
            data.append(row) #all data in a list


    heather=data[0]
    data.pop(0)
    data=np.array(data)


    y=data[:,-1]
    tribute=data[:,tribute_index]
    f1.close()

    return y ,tribute


## Majority vote
def majority_vote(y,option):
    Outcometype=np.unique(y)
    s=0
    ss=0

    for i in range(len(option)):
        if option[i]== Outcometype[0]:
            s+=1
        else:
            ss+=1
    if s>ss:
        vote1=Outcometype[0]
    elif s<ss:
        vote1=Outcometype[1]
    else:
        vote1=Outcometype[np.random.choice([0,1])]

    return vote1


def getLabels_errors(y,tribute,label,vote1,vote2):
    label_clas=[]
    for i in range(len(y)):
        if tribute[i]==label[0]:
            label_clas.append(vote1)
        elif tribute[i]==label[1]:
            label_clas.append(vote2)


    label_clas=np.array(label_clas)
    error=(sum(y!=label_clas))/len(y)

    return  label_clas, error


y ,tribute =openFile(infile,tribute_index) #### TRAINING DATA
y_testing ,tribute_testing =openFile(infile_testing,tribute_index) #### TESTING DATA

label=np.unique(tribute)

option1= y[tribute == label[0]]
option2= y[tribute == label[1]]

vote1= majority_vote(y,option1)
vote2= majority_vote(y,option2)
labels_training, errors_training = getLabels_errors(y,tribute,label,vote1,vote2)
label_testing, errors_testing = getLabels_errors(y_testing,tribute_testing,label,vote1,vote2)

outfile = metrics_out #UPDATE NAME
f2=open(outfile,"w")
text=['error(train): ' ,str(errors_training), "\n",'error(test): ', str(errors_testing )]
f2.writelines(text)
f2.close()

outfile = train_out
f2=open(outfile,"w")
text=labels_training.tolist()
f2.writelines('%s\n' % labelx for labelx in text)
f2.close()


outfile = test_out
f2=open(outfile,"w")
text=label_testing.tolist()
f2.writelines('%s\n' % labelx for labelx in text)
f2.close()
