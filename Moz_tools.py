import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import PCA
import math
from fancyimpute import KNN
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


class Plots():

    def __init__(self, df=0):
        if df != 0:
            self.df = df
            self.shape = df.shape
            self.cat = df.dtypes[df.dtypes == object].index
            self.num = df.dtypes[df.dtypes != object].index
            self.y = df.columns[-1]

    def distribution(self, cat=0, num=0):
        if cat != 0:
            self.cat = cat
        if num != 0:
            self.num = num
        for i in range(len(self.cat)):
            sns.countplot(x=self.df[self.cat].columns[i], data=self.df)
            plt.show()
        for i in self.df[self.num].columns:
            self.df[i].plot(kind='density', title=i)
            plt.show()

    def unique_ratio(self):
        plt.figure(figsize=(20, 10))
        (self.df.apply(lambda x: x.unique().shape[0], axis=0) / self.df.shape[0])\
            .plot(kind='bar', title='unique ratio')

    def na_ratio(self, rot=45, threshold=0):
        tmp0 = self.df.isna().sum() / self.df.shape[0]
        tmp1 = tmp0[tmp0 > threshold]
        self.df[tmp1.index].isnull().sum().plot(kind='bar', rot=rot, title='number of missing values')

    def correlations_to_y(self, y=0, num=0, cat=0, threshold=0.6):
        if y != 0:
            self.y = y
        if num != 0:
            self.num = num
        if cat != 0:
            self.cat = cat
        tmp_ = []
        for i in self.num:
            tmp_ += [self.df[self.y].corr(self.df[i])]
        cor = pd.Series(tmp_, index=self.num)
        cor[abs(cor) > threshold].plot(kind="barh")
        plt.show()
        for i in self.cat:
            data = pd.concat([self.df[self.y], self.df[i]], axis=1)
            sns.boxplot(x=i, y=self.y, data=data)
            plt.show()

    def correlation_heat_map(self, threshold=0, method='pearson', show=True):
        corr = self.df.corr(method=method)
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

    def __init__(self, data=0):
        if data != 0:
            self.df = data
            self.cat = df.dtypes[df.dtypes == object].index
            self.num = df.dtypes[df.dtypes != object].index

    def fill_na(self, data, fill_zero, fill_mode, fill_knn, k, fill_value, value='None'):
        if data != 0:
            self.df = data
        for i in fill_zero:
            self.df[[i]] = self.df.fillna(df[i].fillna(value=0))
        for i in fill_mode:
            self.df[[i]] = self.df.fillna(df[i].fillna(value=df[i].mode()[0]))
        for i in fill_knn:
            self.df[[i]] = KNN(k=k).fit_transform(self.df[i])
        for i in fill_value:
            self.df[[i]] = self.df.fillna(df[i].fillna(value=value))

    def fill_outliers(self, data=0, cols='None'):
        if data != 0:
            self.df = data
        if cols == 'None':
            cols = self.num
        for col in cols:
            scaler_ = StandardScaler()
            scaler_.fit(self.df[[col]])
            tmp_ = scaler_.transform(self.df[[col]])
            maxv = tmp_.mean() + 3 * tmp_.std()
            minv = tmp_.mean() - 3 * tmp_.std()
            for i in range(len(tmp_)):
                if tmp_[i] < minv:
                    tmp_[i] = minv
                if tmp_[i] > maxv:
                    tmp_[i] = maxv
            self.df[col] = scaler.inverse_transform(tmp_)
        return self.df

    def skewness(self, data=0, num=None, threshold=0.75):
        if data != 0:
            self.df = data
        if num is not None:
            self.num = num
        skewed_feats = self.df[self.num].apply(lambda x: skew(x.dropna()))
        skewed_feats = skewed_feats[skewed_feats > threshold]
        skewed_feats = skewed_feats.index
        self.df[skewed_feats] = np.log1p(self.df[skewed_feats])


class FeatureEngineering():

    def __init__(self, data=0):
        if data != 0:
            self.df = data

    def dimension_reduction_num(self, cols, new_name, data=0, drop=True, n=1):
        if data != 0:
            self.df = data
        pca = PCA(n_components=n)
        new_col = pca.fit_transform(self.df[cols])
        if drop:
            self.df.drop(columns=cols, inplace=True)
        self.df[new_name] = new_col
        return self.df

    def dimension_reduction_cat(self, cols, new_name, data=0, drop=True, factorize=True):
        if data != 0:
            self.df = data
        if factorize:
            for i in cols:
                self.df[i] = pd.factorize(self.df[i])[0]
        tmp_ = 1
        for i in cols:
            tmp_ *= self.df[i]
        self.df[new_name] = tmp_
        if drop:
            self.df.drop(columns=cols, inplace=True)
        return self.df

    def extra_pow(self, cols, pow=3, data=0):
        if data != 0:
            self.df = data
        newcol_ = []
        for col in cols:
            for i in range(2, pow+1):
                newcol_ += [col+str(i)]
        j, k = 2, 0
        for i in range(len(newcol_)):
            self.df[newcol_[i]] = math.pow(self.df[cols[k]], j)
            if j < pow:
                j += 1
            else:
                k += 1
                j = 2


if __name__ == '__main__':
    df = pd.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)), columns=['a', 'b', 'c', 'd', 'e'])
    tmp = Plots(df)

