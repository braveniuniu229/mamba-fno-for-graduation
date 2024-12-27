from sklearn.multioutput import MultiOutputRegressor
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, DotProduct, Matern
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np

import torch
# SVR Regression
def svr_regression(train_inputs, train_outputs, val_inputs, val_outputs):
    X = np.concatenate([train_inputs, val_inputs], axis=0)
    y = np.concatenate([train_outputs, val_outputs], axis=0)
    ps = PredefinedSplit(
        test_fold=np.concatenate([-1 * np.ones((train_inputs.shape[0], 1)), np.zeros((val_inputs.shape[0], 1))],
                                 axis=0)
    )
    tuned_parameter = [
        {'estimator__kernel': ['rbf'],
         'estimator__epsilon': [0.00001, 0.0002, 0.0005, 0.0001, 0.005, 0.001]
         }
    ]
    svr = MultiOutputRegressor(svm.SVR())
    svr_c = GridSearchCV(estimator=svr, param_grid=tuned_parameter, cv=ps, n_jobs=4)
    svr_c.fit(X, y)
    kernel, epsilon = svr_c.best_params_['estimator__kernel'], svr_c.best_params_['estimator__epsilon']
    print('The optimal parameters are: \n kernel:', kernel, 'epsilon:', epsilon)

    svr = MultiOutputRegressor(svm.SVR(kernel=kernel, epsilon=epsilon))
    svr.fit(train_inputs, train_outputs)
    return svr


# Random Forest Regression
def rf_regression(train_inputs, train_outputs, val_inputs, val_outputs):
    X = np.concatenate([train_inputs, val_inputs], axis=0)
    y = np.concatenate([train_outputs, val_outputs], axis=0)
    ps = PredefinedSplit(
        test_fold=np.concatenate([-1 * np.ones((train_inputs.shape[0], 1)), np.zeros((val_inputs.shape[0], 1))],
                                 axis=0)
    )
    tuned_parameter = [{'n_estimators': [100, 300, 500, 1000]}]
    rfr = RandomForestRegressor()
    rfr_c = GridSearchCV(estimator=rfr, param_grid=tuned_parameter, cv=ps, n_jobs=4)
    rfr_c.fit(X, y)
    n_estimators = rfr_c.best_params_['n_estimators']
    print('The optimal parameters are: \n n_estimators:', n_estimators)

    rfr = RandomForestRegressor(n_estimators=n_estimators)
    rfr.fit(train_inputs, train_outputs)
    return rfr


# Gaussian Process Regression
def gp_regression(train_inputs, train_outputs, val_inputs, val_outputs):
    X = np.concatenate([train_inputs, val_inputs], axis=0)
    y = np.concatenate([train_outputs, val_outputs], axis=0)
    ps = PredefinedSplit(
        test_fold=np.concatenate([-1 * np.ones((train_inputs.shape[0], 1)), np.zeros((val_inputs.shape[0], 1))],
                                 axis=0)
    )
    tuned_parameter = [{'kernel': [RationalQuadratic(length_scale_bounds=(1e-5, 1e5), alpha_bounds=(1e-5, 1e5)),
                                   RBF(length_scale_bounds=(1e-5, 1e5)),
                                   DotProduct(sigma_0_bounds=(1e-5, 1e5)),
                                   Matern(length_scale_bounds=(1e-5, 1e5))]}]
    gpr = GaussianProcessRegressor()
    gpr_c = GridSearchCV(estimator=gpr, param_grid=tuned_parameter, cv=ps, n_jobs=4)
    gpr_c.fit(X, y)
    kernel = gpr_c.best_params_['kernel']
    print('The optimal parameters are: \n kernel:', kernel)

    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(train_inputs, train_outputs)
    return gpr

if __name__ == "__main__":
    train_input = np.random.randn(100, 16)  # 100个训练样本，每个有16个特征
    train_output = np.random.rand(100, 20000)  # 对应100个样本，每个有20000个输出目标
    test_input = np.random.randn(51, 16)  # 51个测试样本
    test_output = np.random.randn(51, 20000)  # 对应的测试输出
    svr_model = svr_regression(train_input, train_output, train_input, train_output)
    svr_predictions = svr_model.predict(test_input)
    print('SVR Predictions Shape:', svr_predictions.shape)

    # Random Forest 模型训练与预测
    rf_model = rf_regression(train_input, train_output, train_input, train_output)
    rf_predictions = rf_model.predict(test_input)
    print('Random Forest Predictions Shape:', rf_predictions.shape)

    # Gaussian Process 模型训练与预测
    gp_model = gp_regression(train_input, train_output, train_input, train_output)
    gp_predictions = gp_model.predict(test_input)
    print('Gaussian Process Predictions Shape:', gp_predictions.shape)