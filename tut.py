from sklearn import svm
from sklearn import datasets


digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100.)

clf.fit(digits.data[:-10], digits.target[:-10])

print clf.predict(digits.data[-5])
print digits.target[-5]
