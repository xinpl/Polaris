from .common import *
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score

def get_num_leaves(tree):
	ret = 0
	for n1,n2 in zip(tree.children_left, tree.children_right):
		if n1 == n2:
			ret += 1
	return ret

train,test,val = get_train_val_test()

train_X, train_Y = get_X_Y(train)
val_X, val_Y = get_X_Y(val)
test_X, test_Y = get_X_Y(test)

train_Y = onehot_to_bin(train_Y)
val_Y = onehot_to_bin(val_Y)
test_Y = onehot_to_bin(test_Y)

clf = tree.DecisionTreeClassifier()
clf.fit(train_X,train_Y)

preds = clf.predict(test_X)

print(clf.tree_.max_depth)
print(accuracy_score(test_Y, preds))

train_stats = []
test_stats = []

for i in range(50):
	clf = tree.DecisionTreeClassifier(min_samples_split = 2, max_depth=i+1)
	clf.fit(train_X,train_Y)
	tree_depth = clf.tree_.max_depth
	num_nodes = clf.tree_.node_count
	num_leaves = get_num_leaves(clf.tree_)
	print("Tree depth: {}, # nodes: {}, # leaves: {}".format(tree_depth, num_nodes, num_leaves))
	preds = clf.predict(train_X)
	acc = accuracy_score(train_Y,preds)
	f1 = f1_score(train_Y,preds)	
	train_stats.append((tree_depth, num_nodes, num_leaves, acc, f1))
	print("Training accuracy: {}, F1: {}".format(acc, f1))
	preds = clf.predict(test_X)
	acc = accuracy_score(test_Y,preds)
	f1 = f1_score(test_Y,preds)	
	print("Testing accuracy: {}, F1: {}".format(acc, f1))
	test_stats.append((tree_depth, num_nodes, num_leaves, acc, f1))

tf = open("./proof_train.csv", 'w')

for t in train_stats:
	tf.write("{},{},{},{},{}\n".format(t[0],t[1],t[2],t[3],t[4]))

tf.flush()
tf.close()

tf = open("./proof_test.csv", 'w')

for t in test_stats:
	tf.write("{},{},{},{},{}\n".format(t[0],t[1],t[2],t[3],t[4]))

tf.flush()
tf.close()
