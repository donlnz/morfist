import numpy as np
import scipy.stats

class MixedSplitter(object):
	def __init__(self,
				 x,
				 y,
				 max_features='sqrt',
				 min_samples_leaf=5,
				 choose_split='mean',
				 class_targets=None):
		self.n_train = x.shape[0]
		self.n_features = x.shape[1]
		self.n_targets = y.shape[1]
		self.class_targets = class_targets if class_targets else []
		self.max_features = max_features
		self.min_samples_leaf = min_samples_leaf
		self.root_impurity = self.__impurity_node(y)
		self.choose_split = choose_split

	def split(self, x, y):
		if self.max_features == 'sqrt':
			self.max_features = int(np.ceil(np.sqrt(self.n_features)))

		return self.__find_best_split(x, y)

	def __find_best_split(self, x, y):
		if x.shape[0] <= self.min_samples_leaf:
			return None, None, np.inf

		best_f = None
		best_v = None
		best_imp = -np.inf

		try_features = np.random.choice(
			np.arange(self.n_features),
			self.max_features,
			replace=False
		)

		for f in try_features:
			values = np.unique(x[:, f])
			if values.size < 2:
				continue

			values = (values[:-1] + values[1:]) / 2

			#random value subsampling
			values = np.random.choice(values, min(2, values.size))
			for v in values:
				imp = self.__try_split(x, y, f, v)
				if imp > best_imp:
					best_f, best_v, best_imp = f, v, imp

		return best_f, best_v, best_imp

	def __try_split(self, x, y, f, t):
		l_idx = x[:, f] <= t
		r_idx = x[:, f] > t

		return self.__impurity_split(y, y[l_idx, :], y[r_idx, :])

	def __impurity_node(self, y):
		def impurity_class(y):
			y = y.astype(int)
			freq = np.bincount(y) / y.size
			freq = freq[freq != 0]
			return 0 - np.array([f * np.log2(f) for f in freq]).sum()

		def impurity_reg(y):
			if np.unique(y).size < 2:
				return 0

			n_bins = 100
			freq, _ = np.histogram(y, bins=n_bins, density=True)
			proba = (freq + 1) / (freq.sum() + n_bins)
			bin_width = (y.max() - y.min()) / n_bins

			return 0 - bin_width * (proba * np.log2(proba)).sum()

		delta = 0.0001
		imp = np.zeros(self.n_targets)
		for i in range(self.n_targets):
			if i in self.class_targets:
				imp[i] = impurity_class(y[:, i]) + delta
			else:
				imp[i] = impurity_reg(y[:, i]) + delta
		return imp

	def __impurity_split(self, y, y_left, y_right):
		n_parent = y.shape[0]
		n_left = y_left.shape[0]
		n_right = y_right.shape[0]

		if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
			return np.inf
		else:
			imp_left = self.__impurity_node(y_left) / self.root_impurity
			imp_right = self.__impurity_node(y_right) / self.root_impurity
			imp_parent = self.__impurity_node(y) / self.root_impurity

			gain_left = (n_left / n_parent) * (imp_parent - imp_left)
			gain_right = (n_right / n_parent) * (imp_parent - imp_right)
			gain = gain_left + gain_right

			#imp = (imp_left + imp_right) / imp_parent

			if self.choose_split == 'mean':
				return gain.mean()
			else:
				return gain.max()


class MixedRandomTree(object):
	def __init__(self,
				 max_features='sqrt',
				 min_samples_leaf=5,
				 choose_split='mean',
				 class_targets=None):
		self.min_samples_leaf = min_samples_leaf
		self.max_features = max_features
		self.class_targets = class_targets if class_targets else []
		self.choose_split = choose_split

	def fit(self, x, y):
		if y.ndim == 1:
			y = y.reshape((y.size, 1))

		self.n_targets = y.shape[1]

		self.splitter = MixedSplitter(x,
									  y,
									  self.max_features,
									  self.min_samples_leaf,
									  self.choose_split,
									  self.class_targets)

		split_f = []
		split_t = []
		leaf_value = []
		l_child = []
		r_child = []
		n_i = []

		split_queue = [(x, y)]
		i = 0
		while len(split_queue) > 0:
			next_x, next_y = split_queue.pop(0)

			leaf_value.append(self._make_leaf(next_y))
			n_i.append(next_y.shape[0])

			f, t, imp = self.splitter.split(next_x, next_y)

			split_f.append(f)
			split_t.append(t)
			if f:
				l_child.append(i + len(split_queue) + 1)
				r_child.append(i + len(split_queue) + 2)
			else:
				l_child.append(None)
				r_child.append(None)

			if f:
				l_idx = next_x[:, f] <= t
				r_idx = next_x[:, f] > t

				split_queue.append((next_x[l_idx, :], next_y[l_idx, :]))
				split_queue.append((next_x[r_idx, :], next_y[r_idx, :]))

			i += 1

		self.f = np.array(split_f)
		self.t = np.array(split_t)
		self.v = np.array(leaf_value)
		self.l = np.array(l_child)
		self.r = np.array(r_child)
		self.n = np.array(n_i)

	def _make_leaf(self, y):
		y_ = np.zeros(self.n_targets)
		for i in range(self.n_targets):
			if i in self.class_targets:
				y_[i] = np.argmax(np.bincount(y[:, i].astype(int)))
			else:
				y_[i] = y[:, i].mean()
		return y_

	def predict(self, x):
		n_test = x.shape[0]
		pred = np.zeros((n_test, self.n_targets))

		def traverse(x, test_idx, node_idx):
			if test_idx.size < 1:
				return

			if not self.f[node_idx]:
				pred[test_idx, :] = self.v[node_idx]
			else:
				l_idx = x[:, self.f[node_idx]] <= self.t[node_idx]
				r_idx = x[:, self.f[node_idx]] > self.t[node_idx]

				traverse(x[l_idx, :], test_idx[l_idx], self.l[node_idx])
				traverse(x[r_idx, :], test_idx[r_idx], self.r[node_idx])

		traverse(x, np.arange(n_test), 0)
		return pred

	def print(self):
		def print_l(level, i):
			if self.f[i]:
				print('\t' * level + '[{} <= {}]:'.format(self.f[i], self.t[i]))
				print_l(level + 1, self.l[i])
				print_l(level + 1, self.r[i])
			else:
				print('\t' * level + str(self.v[i]) + ' ({})'.format(self.n[i]))

		print_l(0, 0)

class MixedRandomForest(object):
	def __init__(self,
				 n_estimators=10,
				 max_features='sqrt',
				 min_samples_leaf=5,
				 choose_split='mean',
				 class_targets=None):
		self.n_estimators = n_estimators
		self.min_samples_leaf = min_samples_leaf
		self.max_features = max_features
		self.class_targets = class_targets if class_targets else []
		self.choose_split = choose_split

	def fit(self, x, y):
		self.estimators = []

		if y.ndim == 1:
			y = y.reshape((y.size, 1))
		self.n_targets = y.shape[1]

		self.class_labels = {}
		for i in filter(lambda x: x in self.class_targets, range(self.n_targets)):
			self.class_labels[i] = np.unique(y[:, i])

		n_train = x.shape[0]
		for i in range(self.n_estimators):
			m = MixedRandomTree(self.max_features,
								self.min_samples_leaf,
								self.choose_split,
								self.class_targets)
			sample_idx = np.random.choice(np.arange(n_train),
										  n_train,
										  replace=True)

			m.fit(x[sample_idx, :], y[sample_idx, :])
			self.estimators.append(m)

	def predict(self, x):
		n_test = x.shape[0]
		pred = np.zeros((n_test, self.n_targets, self.n_estimators))
		for i, m in enumerate(self.estimators):
			pred[:, :, i] = m.predict(x)

		pred_avg = np.zeros((n_test, self.n_targets))
		for i in range(self.n_targets):
			if i in self.class_targets:
				pred_avg[:, i], _ = scipy.stats.mode(pred[:, i, :].T)
				#pred_avg[:, i] = np.argmax(np.bincount(pred[:, i, :]))
			else:
				pred_avg[:, i] = pred[:, i, :].mean(axis=1)

		return pred_avg

	def predict_proba(self, x):
		n_test = x.shape[0]
		pred = np.zeros((n_test, self.n_targets, self.n_estimators))
		for i, m in enumerate(self.estimators):
			pred[:, :, i] = m.predict(x)

		pred_avg = np.zeros((n_test, self.n_targets), dtype=object)
		for i in range(self.n_targets):
			if i in self.class_targets:
				for j in range(n_test):
					freq = np.bincount(pred[j, i, :].T.astype(int),
									   minlength=self.class_labels[i].size)
					pred_avg[j, i] = freq / self.n_estimators
			else:
				pred_avg[:, i] = pred[:, i, :].mean(axis=1)

		return pred_avg


def acc(y, y_hat):
	return (y.astype(int) == y_hat.astype(int)).sum() / y.size

def rmse(y, y_hat):
	return np.sqrt(((y - y_hat) ** 2).mean())

def cross_validation(model,
					 x,
					 y,
					 folds=10,
					 class_targets=None,
					 class_eval=acc,
					 reg_eval=rmse,
					 verbose=False):
	import copy

	class_targets = class_targets if class_targets else []

	idx = np.random.permutation(x.shape[0])
	fold_size = int(idx.size / folds)

	if y.ndim == 1:
		y = y.reshape((y.size, 1))

	y_hat = np.zeros((idx.size, y.shape[1]))

	for i in range(folds):
		if verbose:
			print('Running fold {} of {} ...'.format(i + 1, folds))

		fold_start = i * fold_size
		fold_stop = min((i + 1) * fold_size, idx.size)

		mask = np.ones(idx.size, dtype=bool)
		mask[fold_start:fold_stop] = 0

		train_idx = idx[mask]
		test_idx = idx[(1 - mask).astype(bool)]

		m = copy.copy(model)
		m.fit(x[train_idx, :], y[train_idx, :])
		y_hat[test_idx, :] = m.predict(x[test_idx, :])

	scores = np.zeros(y.shape[1])
	for i in range(y.shape[1]):
		if i in class_targets:
			scores[i] = class_eval(y[:, i], y_hat[:, i])
		else:
			scores[i] = reg_eval(y[:, i], y_hat[:, i])

	return scores
