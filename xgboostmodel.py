import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ml_metrics import quadratic_weighted_kappa
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support
from collections import namedtuple
from scipy import optimize
import sys


ModelPrediction = namedtuple(
    'ModelPrediction',
    ['ytrain', 'ytest', 'ystrain', 'ystest', 'yhtrain', 'yhtest',
     'qwktrain', 'qwktest', 'precisiontrain', 'precisiontest'])

DataPack = namedtuple('DataPack', 'features labels submission_features')


def add_feature(df, new_feature):
    """
    Adds a new feature to a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame

        DataFrame to which new feature will be added.

    new_feature : dictionary, must contain 'name' and 'func':

        'name' is the name that will be given to the new feature.
        'func' is the function that will be applied to a dataframe row to
        obtain the new feature.
    """
    new_series = df.apply(new_feature['func'], axis=1)
    new_series.name = new_feature['name']
    return df.join(new_series)


def classify_with_cutoffs(yscore, cutoffs):
    """
    Receives a list of seven cutoffs, which will determine the mapping from
    scores to categories. Will be used to convert regression scores into
    one of the seven categories.

    Parameters
    ----------

    predicted_score : array

        Array of predicted scores, which will be mapped onto categories
        according to cutoffs

    cutoffs : array

        Array of length 7 (num_categories - 1).
    """
    assert len(cutoffs) == 7
    cutoffs = np.sort(cutoffs)
    return np.digitize(yscore, cutoffs).astype('int')


def optimize_cutoffs(yscore, ytrue, errorfun=None, *, verbose=False):
    """
    Receives an array of predicted scores, and an array of true values.
    Determines which cutoff values make for the best prediction with
    respect to the true values.

    Parameters
    ----------

    yscore : array

        Array of predicted scores

    ytrue : array

        Array of true  values

    errorfun : func(params, yscore, ytrue) -> error

        Error function that will be used as metric to optimize the cutoffs.

    verbose : bool (optional)

        When true prints the std dev of y-ypred before and after
        optimization.
    """

    yscore = np.asarray(yscore, dtype=np.float64)
    ytrue = np.asarray(ytrue, dtype=np.float64)

    def default_errorfun(p, ysc, ytr):
        """
        Parameters
        ----------

        p : array of 8 cutoff values

        ysc : array of scores [array(double)]

        ytr : array of true labels [array(int)]
        """
        errors = quadratic_weighted_kappa(
            classify_with_cutoffs(ysc, p).astype(np.int64), ytr)
        return 1 - errors

    if errorfun is None:
        errorfun = default_errorfun

    def error(p):
        return errorfun(p, yscore, ytrue)

    cutoffs0 = np.arange(7)+0.5

    just = 15
    if verbose:
        print("{} : {}".format(
                "start error".rjust(just), error(cutoffs0)))

    # xopt, fopt, niter, funcalls, warnflag, allvecs
    pfit = optimize.fmin_powell(error, cutoffs0, xtol=1e-2, ftol=1e-6,
                                maxiter=None, maxfun=None, disp=verbose)

    if verbose:
        print("{} : {}\n".format(
                "final error".rjust(just), error(pfit)))

    return np.sort(pfit)


class XGBoostModel:

    def __init__(self, nfolds=3, datapack=None):
        """
        Initialize the xgboost model by loading the imputed data from file.

        The train and test sets will be split according to the 'train?' column.

        Parameters
        ----------

        nfolds : int

            Number of folds that will be used to setup cross validation


        Class data
        ----------

        self.features, self.labels :

            Features and labeles for all of the labeled data available.

        self.submisison_features :

            Features for the data that needs to be predicted in the submisison.

        self.classif_param :  dict

            A set of default parameters for using the 'multi:softmax'
            objective.

        self.linear_param : dict

            A set of default paramters for using the 'reg:linear' objective.

        self.cross_valid : list(tuple(array, array))

            A list of length equal to nfolds.  Each item represents a cross
            validation fold, and contains a tuple with index arrays for the
            training and testing data in the fold.


        """

        if datapack is None:
            data = pd.read_csv('csvs/data_imputed.csv')

            self.features = data[data['train?']].drop(
                ['train?', 'Id', 'Response'], axis=1)
            self.labels = data[data['train?']]['Response'] - 1

            self.submission_features = data[~data['train?']].drop(
                ['train?', 'Id', 'Response'],
                axis=1)
        else:
            self.features = datapack.features
            self.labels = datapack.labels
            self.submission_features = datapack.submission_features

        self.classif_param = {}
        self.classif_param['objective'] = 'multi:softmax'
        self.classif_param['num_class'] = 8
        self.classif_param['eta'] = 0.1
        self.classif_param['max_depth'] = 6
        self.classif_param['min_child_weight'] = 1
        self.classif_param['subsample'] = 0.5
        self.classif_param['colsample_bytree'] = 0.67
        self.classif_param['silent'] = 1
        self.classif_param['nthread'] = 2

        #
        self.linear_param = {}
        self.linear_param['objective'] = 'reg:linear'
        self.linear_param['eta'] = 0.1
        self.linear_param['max_depth'] = 6
        self.linear_param['min_child_weight'] = 1
        self.linear_param['subsample'] = 0.5
        self.linear_param['colsample_bytree'] = 0.67
        self.linear_param['silent'] = 1
        self.linear_param['nthread'] = 1

        # Setup reusable folds for cross-validation
        self.nfolds = nfolds
        self.cross_valid = list(KFold(len(self.features),
                                      n_folds=self.nfolds,
                                      shuffle=True,
                                      random_state=12))

        # List will store parameters and results every time a
        # model is trained
        self.models = []
        self.scores = []

    def make_split(self, train_size=0.7):
        """
        Splits data into train and test sets.

        Parameters
        ----------
        train_size : float between 0 and 1

            Fraction of data in training set

        Returns
        -------
        xg_train : xgboost.DMatrix
            xgboost representation of the training features and labels

        xg_test : xgboolst.DMatrix
            xgboost representation of the testing features and labels

        ttsplit : tuple with len=4
            ttsplit[0] = Xtrain
            ttsplit[1] = Xtest
            ttsplit[2] = ytrain
            ttsplit[3] = ytest
        """

        if train_size < 1:
            ttsplit = train_test_split(
                self.selected_features,
                self.labels,
                train_size=train_size,
                random_state=17)
        else:
            ttsplit = (self.selected_features, None, self.labels, None)

        xg_train = xgb.DMatrix(ttsplit[0], ttsplit[2])
        xg_test = xgb.DMatrix(ttsplit[1], ttsplit[3])

        return xg_train, xg_test

    def make_cv_split(self, cvindex, returnxgb=True):
        """
        Splits data into training and test sets, according to the reusable
        cross validation folds that were prepared during initialization.

        Parameters
        ----------
        cvindex : int

            Index of cross validation item that will be used to split the
            data
        """

        if cvindex >= self.nfolds:
            raise ValueError('make_cv_split error: ' +
                             'Invalid index {}'.format(cvindex) +
                             'Only {} folds'.format(self.nfolds) +
                             'are setup for cross validation. ')

        idx_train = self.cross_valid[cvindex][0]
        idx_test = self.cross_valid[cvindex][1]

        ttsplit = (
            self.selected_features.values[idx_train],
            self.selected_features.values[idx_test],
            self.labels.values[idx_train],
            self.labels.values[idx_test])

        if not returnxgb:
            return (ttsplit[0], ttsplit[2]), (ttsplit[1], ttsplit[3])

        xg_train = xgb.DMatrix(ttsplit[0], ttsplit[2])
        xg_test = xgb.DMatrix(ttsplit[1], ttsplit[3])

        return xg_train, xg_test

    def predict(self, model, xg_train, xg_test, objective='reg:linear'):
        """
        Parameters
        ----------

        model : xgboost.Booster
            xgboost model ready for making predictions

        xg_train : xgboost.DMatrix
            training data

        xg_test : xgboost.DMatrix
            testing data


        Returns
        -------

        model_prediction : ModelPrediction (named tuple)

        """

        train_score = model.predict(
            xg_train, ntree_limit=model.best_iteration)
        test_score = model.predict(
            xg_test,  ntree_limit=model.best_iteration)

        train_label = np.asarray(xg_train.get_label())
        test_label = np.asarray(xg_test.get_label())

        if objective == 'reg:linear':
            # Cuttofs are optimized here
            best_cuts = optimize_cutoffs(train_score, train_label,
                                         verbose=False)
            train_prediction = classify_with_cutoffs(train_score, best_cuts)
            test_prediction = classify_with_cutoffs(test_score, best_cuts)
        else:
            train_prediction = train_score
            test_prediction = test_score

        train_qwk = quadratic_weighted_kappa(train_label, train_prediction)
        test_qwk = quadratic_weighted_kappa(test_label, test_prediction)

        return ModelPrediction(train_label, test_label,
                               train_score, test_score,
                               train_prediction, test_prediction,
                               train_qwk, test_qwk,
                               precision_score(train_label, train_prediction,
                                               average=None),
                               precision_score(test_label, test_prediction,
                                               average=None)
                               )

    def learn_model(self, fold=0.7, objective='reg:linear', num_round=50,
                    make_plot=True, feature_quantile_cut=0.0,
                    custom_features=None, print_qwk=False, **kwargs):
        """
        Train an xgb ensemble of trees.

        Parameters
        ----------

        fold : Number in the range (0,1) or int in range(len(self.cross_valid))

            If number is a float in the range 0 < fold < 1, it will be
            interpreted as a train_size percentage in a train-test split.

            If number is an integer in the range(len(self.cross_valid)) it
            will be interpreted as an index in the list of reusable cross
            validation forms that were set up during class initialization.

        num_round : int,

            Number of boosting rounds.

        make_plot : bool,

            Make a plot evaluating the model results

        feature_quantile_cut : float in range 0. to 1.

            The quantile cutoff for feature selection.  Selection is based on
            coefficients from a ridge regression peformed previously in R. Only
            features with linear model coefficients in the range above the
            quantile cut will be preserved.

        custom_features : list

            Allows the definition and incorporation of custom features into
            the data set.
        """

        if feature_quantile_cut > 0.0:
            ridge_coefs = pd.read_csv('csvs/R_ridge_coefficients.csv').iloc[0]
            cut = ridge_coefs.quantile(feature_quantile_cut)
            to_drop = list(ridge_coefs[ridge_coefs < cut].index.values)
            if 'Id' in to_drop:
                to_drop.remove('Id')
            self.selected_features = self.features.drop(to_drop, axis=1)
        else:
            self.selected_features = self.features

        if custom_features is not None:
            for cf in custom_features:
                self.selected_features = add_feature(self.selected_features,
                                                     cf)

        if float(fold) > 0 and float(fold) < 1:
            xg_train, xg_test = self.make_split(train_size=fold)
            fold_str = '{:0.2f}'.format(fold)
        elif int(fold) in range(len(self.cross_valid)):
            xg_train, xg_test = self.make_cv_split(int(fold))
            fold_str = 'CV{:d}'.format(fold)
        else:
            raise ValueError("learn_model error: " +
                             "invalid value for fold: {:d}".format(fold))

        if objective == 'reg:linear':
            params = self.linear_param
        elif objective == 'multi:softmax':
            params = self.classif_param
        else:
            raise ValueError("learn_model error: "
                             "{} is not a valid objective".format(objective))

        for key, val in kwargs.items():
            if key in params.keys():
                params[key] = val

        model = xgb.train(params, xg_train, num_round)
        pred = self.predict(model, xg_train, xg_test, objective=objective)

        if print_qwk:
            print("qwktrain = {:0.4f}, qwktest = {:0.4f}".format(
                pred.qwktrain, pred.qwktest))
            sys.stdout.flush()

        self.save_score(model, feature_quantile_cut, custom_features,
                        fold, fold_str,
                        params, num_round, pred,
                        make_plot=make_plot)

    def cv_model(self, **kwargs):
        """
        Learns the model on each of the cross validation folds in
        self.cross_valid
        """

        for i in range(self.nfolds):
            self.learn_model(fold=i, **kwargs)

    def get_scores(self):
        """
        Makes a dataframe with all of the recorded scores.

        Returns
        -------

        df_score : pandas.DataFrame

            Each row represents a fit of the model with certain
            paramters and conditions.
        """

        return pd.DataFrame(self.scores)

    def save_score(self, model, feature_quantile_cut, custom_features,
                   fold, fold_str,
                   params, num_round, pred,
                   make_plot=True):
        """
        Saves the results of the model as a dictionary

        Parameters
        ----------

        model : xgboost.Booster

            The model.

        feature_quantile_cut : float in range 0 to 1

            Cutoff used to select features based on their linear ridge coefs.

        custom_features : list

            List of custom features added to data.

        fold : float or int

            Describes which fold was used to train the model.
            See description for fold_str.

        fold_str : string

            A string that describes how the training and testing
            sets were split. If string is a decimal number, then
            split was a random train-test split.  If string is like
            CV#, then the #th precalculated cv split was used.

        params : dict

            Parameters that were used to fit the model

        num_rounds : int

            number of boosting rounds used

        pred : ModelPrediction

            named tuple with the prediction results

        """

        score = params.copy()
        score['feature_quantile_cut'] = "{:0.2f}".format(feature_quantile_cut)
        if custom_features is not None:
            score['custom_features'] = ', '.join(cf['name']
                                                 for cf in custom_features)
        else:
            score['custom_features'] = 'None'
        score['fold_str'] = fold_str
        score['num_round'] = num_round
        score['train_qwk'] = pred.qwktrain
        score['test_qwk'] = pred.qwktest

        precision, recall, fscore, _ = precision_recall_fscore_support(
            pred.ytrain, pred.yhtrain, average='micro')

        score['train_precision'] = precision
        score['train_recall'] = recall
        score['train_fbetascore'] = fscore

        precision, recall, fscore, _ = precision_recall_fscore_support(
            pred.ytest, pred.yhtest, average='micro')

        score['test_precision'] = precision
        score['test_recall'] = recall
        score['test_fbetascore'] = fscore

        self.models.append((model, fold, pred))
        self.scores.append(score)
        if make_plot:
            self.make_plot_eval(params, num_round, pred)

    def make_plot_eval(self, params, num_round, pred):
        """
        Makes a plot that evaluates a prediction with respect to known labels.
        The plot consists of two panels: freq vs category on the left panel and
        confusion matrix on the right panel.

        Parameters
        ----------

        params : dict

            Dictionary with the model parameters.

        num_round : int

            Number of boosting rounds used during training.

        pred : ModelPrediction

            A named tuple that contains the prediction results.
        """

        # Setup texts for the plot
        text = "{} eta:{:0.2f}, max_depth:{:d}\n".format(
                self.classif_param['objective'],
                self.classif_param['eta'],
                self.classif_param['max_depth'],) + \
            "min_child_weight:{:0.2f}, num_round:{:02d}".format(
                self.classif_param['min_child_weight'],
                num_round)
        ktrain = '${}={:0.4f}$, '.format(
            '\kappa_{{q,\mathrm{{train}}}}',
            pred.qwktrain)
        ktest = '${}={:0.4f}$'.format(
            '\kappa_{{q,\mathrm{{train}}}}',
            pred.qwktest)
        kappa_text = ktrain + ktest

        # Define label and prediction
        y, yhat = pred.ytest, pred.yhtest

        # Make the figure
        fig, ax = plt.subplots(1, 2, figsize=(9., 4.))

        y = pd.Series(y)
        yhat = pd.Series(yhat)

        yhist = y.value_counts()
        yhathist = yhat.value_counts()

        ax[0].scatter(
            yhist.index,
            yhist.values,
            s=40,
            c='blue',
            alpha=0.5,
            label='response')
        ax[0].scatter(
            yhathist.index-0.1,
            yhathist.values,
            s=40,
            c='green',
            alpha=0.5,
            label='prediction')

        ax[0].legend(loc='upper left', prop={'size': 10})
        ax[0].set_xlabel('classification')
        ax[0].set_ylabel('frequency')

        ax[0].text(0.99, 1.01, text, fontsize=9,
                   ha='right', va='bottom', transform=ax[0].transAxes)

        cm = confusion_matrix(y, yhat)
        print(cm)

        im = ax[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax[1].set_xlabel('true label')
        ax[1].set_ylabel('predicted label')
        plt.colorbar(im)

        ax[1].text(0.99, 1.01, kappa_text, fontsize=12,
                   ha='right', va='bottom', transform=ax[1].transAxes)

        fname = 'plots/classification_' + text.replace(
            ':', '').replace(
            ', ', '_').replace(
            ' ', '') + '.png'
        fig.savefig(fname, dpi=150)
