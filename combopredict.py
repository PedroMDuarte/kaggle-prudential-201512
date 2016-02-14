from xgboostmodel import ModelPrediction, classify_with_cutoffs


class ComboPredict:

    def __init__(self, booster):
        """
        Intitialize the combination predictor with an XGBoostModel instance

        Parameters
        ----------

        booster : XGBoostModel

        """

        self.booster = booster

        if (len(self.booster.models) == 0):
            raise ValueError("The XGBoostModel provided does not contain any "
                             "fitted models.")

    def predict_score(self, features, overall_cls_factor):
        """
        Predicts scores for a set of observations.  The score is calculated
        by weighting all of the models present in the XGBoostModel that is
        passed into the class at initialization.

        Note that the scores output by this function need to be coerced to
        category values for a final prediction.

        Parameters
        ----------

        features : array

            Features for which predictions weill be generated

        overall_cls_factor : float

            Relative weight of the classification boosters wrt to the
            regression boosters
        """

        xg_input = xgb.DMatrix(features)

        weighted_preds = []
        norms = []
        for m in zip(self.booster.models, self.booster.scores):
            model, model_fold, model_pred = m[0]
            score = m[1]

            nfeatures = xg_input.num_row()
            X, _ = np.meshgrid(np.arange(8), np.arange(nfeatures))

            if score['objective'] == 'multi:softmax':
                pred_cls = model.predict(xg_input,
                                         ntree_limit=model.best_iteration)
                dummies = pd.get_dummies(pred_cls).values
                weight = model_pred.precisiontrain.reshape(8, 1)

                weighted_pred = np.dot(X * dummies,
                                       overall_cls_factor * weight)
                weighted_preds.append(weighted_pred)

                norm = np.dot(dummies, overall_cls_factor * weight)
                norms.append(norm)

            else:
                reg_pred = model.predict(xg_input,
                                         ntree_limit=model.best_iteration)
                weighted_preds.append(reg_pred.reshape(nfeatures, 1))
                norms.append(np.ones((nfeatures,1)))

        total = np.sum(weighted_preds, axis=0)
        norm = np.sum(np.array(norms), axis=0)

        combo_score = np.squeeze(total / norm)
        return combo_score

    def predict_categories(self, features, overall_cls_factor, cuts):
        """
        Makes a call to predict_scores, and then classifies into categories
        based on the cuts.

        Parameters
        ----------

        features, overall_cls_factor :  see predict_scores for definition.

        cuts : array of length 7

            Cutoffs that will be used to convert scores into categories.
        """

        scores = self.predict_score(features, overall_cls_factor)
        yhcombo = classify_with_cutoffs(scores, cuts)
        return yhcombo
