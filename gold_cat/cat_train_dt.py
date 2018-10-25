from .common import create_batch2, separate_data
import sys
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score

def get_num_leaves(tree):
	ret = 0
	for n1,n2 in zip(tree.children_left, tree.children_right):
		if n1 == n2:
			ret += 1
	return ret

val_size = 10000

test_size = 10000

def main():
    gold, train, val, test = separate_data()
    train_X,train_Y = create_batch2(20000, gold,train, True)
    test_X, test_Y = create_batch2(20000, gold, test, True)
    train_X = np.reshape(train_X, [-1, 512])
    test_X = np.reshape(test_X, [-1, 512])
    train_Y = [y[1] > y[0] for y in train_Y]
    test_Y = [y[1] > y[0] for y in test_Y]


    clf = tree.DecisionTreeClassifier()
    clf.fit(train_X,train_Y)

    preds = clf.predict(test_X)

    print(clf.tree_.max_depth)
    tree.export_graphviz(clf, out_file="./tree.dot")
    print(accuracy_score(test_Y, preds))
    #
    # train_stats = []
    # test_stats = []
    #
    # for i in range(50):
    #     clf = tree.DecisionTreeClassifier(min_samples_split = 2, max_depth=i+1)
    #     clf.fit(train_X,train_Y)
    #     tree_depth = clf.tree_.max_depth
    #     num_nodes = clf.tree_.node_count
    #     num_leaves = get_num_leaves(clf.tree_)
    #     print("Tree depth: {}, # nodes: {}, # leaves: {}".format(tree_depth, num_nodes, num_leaves))
    #     preds = clf.predict(train_X)
    #     acc = accuracy_score(train_Y,preds)
    #     f1 = f1_score(train_Y,preds)
    #     train_stats.append((tree_depth, num_nodes, num_leaves, acc, f1))
    #     print("Training accuracy: {}, F1: {}".format(acc, f1))
    #     preds = clf.predict(test_X)
    #     acc = accuracy_score(test_Y,preds)
    #     f1 = f1_score(test_Y,preds)
    #     print("Testing accuracy: {}, F1: {}".format(acc, f1))
    #     test_stats.append((tree_depth, num_nodes, num_leaves, acc, f1))
    #
    # tf = open("./cat_train.csv", 'w')
    #
    # for t in train_stats:
    #     tf.write("{},{},{},{},{}\n".format(t[0],t[1],t[2],t[3],t[4]))
    #
    # tf.flush()
    # tf.close()
    #
    # tf = open("./cat_test.csv", 'w')
    #
    # for t in test_stats:
    #     tf.write("{},{},{},{},{}\n".format(t[0],t[1],t[2],t[3],t[4]))
    #
    # tf.flush()
    # tf.close()



if __name__ == '__main__':
    main()
