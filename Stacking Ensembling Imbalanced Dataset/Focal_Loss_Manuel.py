import numpy as np
import lightgbm as lgb
import math
from sklearn.metrics import f1_score
from scipy.misc import derivative


def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
	a,g = alpha, gamma
	y_true = dtrain.label
	def fl(x,t):
		p = 1/(1+np.exp(-x))
		return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
	partial_fl = lambda x: fl(x, y_true)
	grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
	hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
	return grad, hess


def focal_loss_lgb_f1_score(preds, lgbDataset):
  preds = math.sigmoid(preds)
  binary_preds = [int(p>0.5) for p in preds]
  y_true = lgbDataset.get_label()
  return 'f1', f1_score(y_true, binary_preds), True

def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
  a,g = alpha, gamma
  y_true = dtrain.label
  p = 1/(1+np.exp(-y_pred))
  loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
  # (eval_name, eval_result, is_higher_better)
  return 'focal_loss', np.mean(loss), False




x_train = np.load('x_train.npy')
x_vali = np.load('x_vali.npy')
y_train = np.load('y_train.npy')
y_vali = np.load('y_vali.npy')




focal_loss = lambda x,y: focal_loss_lgb(x, y, 0.25, 2.)
eval_error = lambda x,y: focal_loss_lgb_eval_error(x, y, 0.25, 2.)
lgbtrain = lgb.Dataset(x_train, y_train, free_raw_data=True)
lgbeval = lgb.Dataset(x_vali, y_vali)
params  = {'learning_rate':0.1, 'num_boost_round':10}
model = lgb.train(params, lgbtrain, valid_sets=[lgbeval], fobj=focal_loss, feval=eval_error )


