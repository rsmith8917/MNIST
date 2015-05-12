import pandas
from sklearn import svm
from sklearn.externals import joblib

train = pandas.read_csv('/home/robert/Downloads/train.csv')
test = pandas.read_csv('/home/robert/Downloads/test.csv')

labels = train.label
images = train[train.columns[1:]]

clf = svm.LinearSVC(C=0.00000001)

clf.fit(images, labels)

joblib.dump(clf, '/home/robert/svm.pkl')

test_output = pandas.DataFrame({'ImageId': range(1, len(test)+1),
                                'Label': clf.predict(test)})

test_output.to_csv('/home/robert/svm.csv', index=False)




