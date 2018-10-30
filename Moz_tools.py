import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import PCA
import math
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from scipy.special import boxcox1p
from fancyimpute import KNN
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

class Plots():

    def __init__(self, data=None):
        if data is not None:
            self.df = data
            self.shape = data.shape
            self.cat = data.dtypes[data.dtypes == object].index
            self.num = data.dtypes[data.dtypes != object].index
            self.y = data.columns[-1]

    def distribution(self, df=None, cat=0, num=0):
        if df is None:
            df = self.df
        if cat == 0:
            cat = self.cat
        if num == 0:
            num = self.num
        for i in range(len(cat)):
            sns.countplot(x=df[cat].columns[i], data=df)
            plt.show()
        for i in df[num].columns:
            df[i].plot(kind='density', title=i)
            plt.show()

    def unique_ratio(self, df=None, threshold=0):
        if df is None:
            df = self.df
        plt.figure(figsize=(20, 10))
        df = df.apply(lambda x: x.unique().shape[0], axis=0) / df.shape[0]
        df[df > threshold].plot(kind='bar', title='unique ratio')

    def na_ratio(self, df=None, cols=None, rot=45, threshold=0):
        if df is None:
            df = self.df
        if cols is None:
            cols = df.columns
        tmp0 = df.isna().sum() / df.shape[0]
        tmp1 = tmp0[tmp0 > threshold]
        df[cols][tmp1.index].isnull().sum().plot(kind='bar', rot=rot, title='number of missing values')

    def correlations_to_y(self, df=None, y=0, num=0, cat=0, threshold=0.6):
        if df is None:
            df = self.df
        if y == 0:
            y = self.y
        if num == 0:
            num = self.num
        if cat == 0:
            cat = self.cat
        tmp_ = []
        for i in num:
            tmp_ += [df[y].corr(df[i])]
        cor = pd.Series(tmp_, index=num)
        cor[abs(cor) > threshold].plot(kind="barh")
        plt.show()
        for i in cat:
            data = pd.concat([df[y], df[i]], axis=1)
            sns.boxplot(x=i, y=y, data=data)
            plt.show()

    def correlation_scatter(self, df=None, columns=None):
        if df is None:
            df = self.df
        if columns is None:
            columns = self.num
        sns.set()
        sns.pairplot(df[columns], size=2.5)
        plt.show()

    def correlation_heat_map(self, df=None, threshold=0, method='pearson', show=True):
        if df is None:
            df = self.df
        corr = df.corr(method=method)
        c = corr[abs(corr) > threshold]
        c = c[c != 1]
        c.dropna(how='all', axis=1, inplace=True)
        c.dropna(how='all', inplace=True)
        mask = np.zeros_like(c, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        fig, ax = plt.subplots(figsize=(10, 10))
        colormap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(c, mask=mask, cmap=colormap, annot=show, fmt=".2f")
        plt.xticks(range(len(c.columns)), c.columns)
        plt.yticks(range(len(c.columns)), c.columns)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        print('Confusion matrix, without normalization')
        print(cm)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


class DataCleaner():

    def __init__(self, df=None):
        if df is not None:
            self.df = df
            self.cat = self.df.dtypes[self.df.dtypes == object].index
            self.num = self.df.dtypes[self.df.dtypes != object].index

    def fill_na(self, df=None, fill_zero=None, fill_mode=None, fill_knn=None, k=None, fill_value=None, value='None'):
        if df is None:
            df = self.df
        if fill_zero is not None:
            for i in fill_zero:
                df[[i]] = df[[i]].fillna(value=0)
        if fill_mode is not None:
            for i in fill_mode:
                df[[i]] = df[[i]].fillna(value=df[i].mode()[0])
        if fill_knn is not None:
            for i in fill_knn:
                df[[i]] = KNN(k=k).fit_transform(df[[i]])
        if fill_value is not None:
            for i in fill_value:
                df[[i]] = df[[i]].fillna(value=value)
        return df

    def fill_outliers(self, df=None, cols=None, method='Standardize', replace=True):
        if df is None:
            df = self.df
        if cols is None:
            cols = self.num
        if method == 'Standardize':
            for col in cols:
                scaler_ = StandardScaler()
                scaler_.fit(df[[col]])
                tmp_ = scaler_.transform(df[[col]])
                maxv = tmp_.mean() + 3 * tmp_.std()
                minv = tmp_.mean() - 3 * tmp_.std()
                if replace:
                    for i in range(len(tmp_)):
                        if tmp_[i] < minv:
                            tmp_[i] = minv
                        if tmp_[i] > maxv:
                            tmp_[i] = maxv
                else:
                    tmp_ = tmp_[tmp_ > minv][tmp_ < maxv]
                df[col] = scaler_.inverse_transform(tmp_)
        elif method == 'IQR':
            for col in cols:
                tmp_ = df[col]
                IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
                m = df[col].mean()
                maxv = m+3*IQR
                minv = m-3*IQR
                if replace:
                    for i in range(len(tmp_)):
                        if tmp_[i] < minv:
                            tmp_[i] = minv
                        if tmp_[i] > maxv:
                            tmp_[i] = maxv
                else:
                    df[col] = tmp_[tmp_ > minv][tmp_ < maxv]
        return df

    def skewness(self, df=None, num=None, method='box-cox', lamd=0.16, threshold=0.75):
        if df is None:
            df = self.df
        if num is None:
            num = self.num
        skewed_feats = df[num].apply(lambda x: skew(x.dropna()))
        skewed_feats = skewed_feats[skewed_feats > threshold]
        skewed_feats = skewed_feats.index
        if method == 'log1p':
            df[skewed_feats] = np.log1p(df[skewed_feats])
        if method == 'box-cox':
            df[skewed_feats] = boxcox1p(df[skewed_feats], lamd)
        return df


class FeatureEngineering():

    def __init__(self, df=None):
        if df is not None:
            self.df = df

    def dimension_reduction_num(self, cols, new_name, df=None, drop=True, n=1):
        if df is None:
            df = self.df
        pca = PCA(n_components=n)
        new_col = pca.fit_transform(df[cols])
        if drop:
            df.drop(columns=cols, inplace=True)
        df[new_name] = new_col
        return df

    def dimension_reduction_cat(self, cols, new_name, df=None, drop=True, factorize=True):
        if df is None:
            df = self.df
        if factorize:
            for i in cols:
                df[i] = pd.factorize(df[i])[0]
        tmp_ = 1
        for i in cols:
            tmp_ *= df[i]
        df[new_name] = tmp_
        if drop:
            df.drop(columns=cols, inplace=True)
        return df

    def extra_pow(self, cols, pow=3, df=None):
        if df is None:
            df = self.df
        for p in range(2, pow+1):
            for col in cols:
                newcol_ = col + str(p)
                tmp_ = []
                for i in range(len(df[col])):
                    tmp_.append(math.pow(df[col][i], p))
                df[newcol_] = tmp_
        return df


def rmse_cv(model, train_x, train_y, cv=10):
    rmse = np.sqrt(-cross_val_score(model, train_x, train_y, scoring="neg_mean_squared_error", cv=cv))
    return rmse


class FeatureSelection():
    def __init__(self, train_x=None, train_y=None, test_x=None):
        if train_x is not None:
            self.train_x = train_x
        if train_y is not None:
            self.train_y = train_y
        if test_x is not None:
            self.test_x = test_x

    def lassocv(self, alphas=[1, 0.1, 0.001, 0.0005], train_x=None, train_y=None, cv=10):
        if train_x is None:
            train_x = self.train_x
        if train_y is None:
            train_y = self.train_y
        model_lasso = LassoCV(alphas=alphas, cv=cv).fit(train_x, train_y)
        coef = pd.Series(model_lasso.coef_, index=train_x.columns)
        print('rmse score:', rmse_cv(model_lasso, train_x=train_x, train_y=train_y, cv=cv).mean())
        return coef

    def ridgecv(self, alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10], train_x=None, train_y=None, cv=10):
        if train_x is None:
            train_x = self.train_x
        if train_y is None:
            train_y = self.train_y
        cv_ridge = [rmse_cv(Ridge(alpha=alpha), train_x, train_y, cv=cv).mean()
                    for alpha in alphas]
        cv_ridge = pd.Series(cv_ridge, index=alphas)
        cv_ridge.plot(title="Validation - Ridge")
        plt.xlabel("alpha")
        plt.ylabel("rmse")
        plt.show()
        print('rmse score:', cv_ridge.min())
        r = Ridge(cv_ridge.idxmin())
        r.fit(train_x, train_y)
        coef = pd.Series(r.coef_, index=train_x.columns)
        return coef



if __name__ == '__main__':
    df = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)), columns=['a', 'b', 'c', 'd', 'e'])
    tmp = Plots(df)
