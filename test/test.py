from morfist import MixedRandomForest, cross_validation
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston, load_breast_cancer
import numpy as np

# Config
n_trees = 11

# Data
x_reg, y_reg = load_boston(return_X_y=True)
x_cls, y_cls = load_breast_cancer(return_X_y=True)
x_mix_1, y_mix_1 = x_reg, np.vstack([y_reg, y_reg < y_reg.mean()]).T
x_mix_2, y_mix_2 = x_cls, np.vstack([y_cls, y_cls]).T

def test_class():
	cls_rf = MixedRandomForest(
		n_estimators=n_trees,
		min_samples_leaf=1,
		class_targets=[0]
	)

	cls_skrf = RandomForestClassifier(n_estimators=n_trees)
	cls_scores = cross_validation(
		cls_rf,
		x_cls,
		y_cls,
		class_targets=[0],
		folds=10
	)

	scores = cross_val_score(
		cls_skrf,
		x_cls,
		y_cls
	)

	print('Classification: ')
	print('\tmorfist (accuracy): {}'.format(cls_scores.mean()))
	print('\tscikit-learn (accuracy): {}'.format(scores.mean()))

def test_reg():
	reg_rf = MixedRandomForest(
		n_estimators=n_trees,
		min_samples_leaf=5
	)

	reg_skrf = RandomForestRegressor(n_estimators=n_trees)
	reg_scores = cross_validation(
		reg_rf,
		x_reg,
		y_reg,
		folds=10
	)

	scores = cross_val_score(
		reg_skrf,
		x_reg,
		y_reg,
		scoring='neg_mean_squared_error'
	)

	print('Regression: ')
	print('\tmorfist (rmse): {}'.format(reg_scores.mean()))
	print('\tscikit-learn (rmse): {}'.format(np.sqrt(-scores.mean())))

def test_mix_1():
	mix_rf = MixedRandomForest(
		n_estimators=n_trees,
		min_samples_leaf=5,
		class_targets=[1]
	)

	mix_scores = cross_validation(
		mix_rf,
		x_mix_1,
		y_mix_1,
		folds=10,
		class_targets=[1]
	)
	print('Mixed output: ')
	print('\ttask 1 (original) (rmse): {}'.format(mix_scores[0]))
	print('\ttask 2 (additional) (accuracy): {}'.format(mix_scores[1]))

def test_mix_2():
	mix_rf = MixedRandomForest(
		n_estimators=n_trees,
		min_samples_leaf=1,
		class_targets=[0]
	)

	mix_scores = cross_validation(
		mix_rf,
		x_mix_2,
		y_mix_2,
		folds=10,
		class_targets=[0]
	)
	print('Mixed output: ')
	print('\ttask 1 (original) (accuracy): {}'.format(mix_scores[0]))
	print('\ttask 2 (additional) (rmse): {}'.format(mix_scores[1]))

if __name__ == '__main__':
	test_class()
	print('')
	test_reg()
	print('')
	test_mix_1()
	print('')
	test_mix_2()