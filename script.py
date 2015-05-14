import pandas
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV

train = pandas.read_csv('/home/robert/Downloads/train.csv')
test = pandas.read_csv('/home/robert/Downloads/test.csv')

labels = train.label
images = train[train.columns[1:]]

clf = svm.SVC(C=4.64158883361277752044, gamma=0.00000035938136638046)

param_grid = {"C": np.logspace(-1, 2, 10),
              "gamma": np.logspace(-15, -4, 10)}

# gs = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=2)

# N = 15000

clf.fit(images, labels)

joblib.dump(clf, '/home/robert/svm.pkl')

# # print diagnostic information to the user and grab the
# # best model
# print "best score: %0.3f" % (gs.best_score_)
# print "SVM PARAMETERS"
# bestParams = gs.best_estimator_.get_params()

# # loop over the parameters and print each of them out
# # so they can be manually set
# for p in sorted(param_grid.keys()):
#     print "\t %s: %0.20f" % (p, bestParams[p])

test_output = pandas.DataFrame({'ImageId': range(1, len(test)+1),
                                'Label': clf.predict(test)})

test_output.to_csv('/home/robert/svm.csv', index=False)




