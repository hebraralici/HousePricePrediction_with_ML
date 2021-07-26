##################################################################
# House Prices Prediction with Machine Learning Algorithms
##################################################################

#################
# İş Problemi
#################
"""
Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak, farklı tipteki
evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi gerçekleştirilmek istenmektedir.
"""

#######################
# Veri Seti Hikayesi
#######################
"""
- Ames, Lowa’daki konut evlerinden oluşan bu veriseti içerisinde 79 açıklayıcı 
değişken bulunduruyor.
- Veri setinde  train ve test olmak üzere iki farklı csv dosyası vardır.
- Test veri setinde ev fiyatları boş bırakılmış olup, bu değerleri sizin tahmin etmeniz 
beklenmektedir.
"""

###################
# Değişkenler
###################
"""
- 1460 gözlem
- 38 Sayısal değişken
- 43 Kategorik değişken
- Değişken tanımlarına aşağıdaki linkten ulaşılabilir:
https://docs.google.com/spreadsheets/d/1k2mj1zjCDkf60YOLArAI8XfrkjMa5sxbRQNZHBe1xC8/edit#gid=1166040824

"""
################
# GÖREV
################
""" Elimizdeki veri seti üzerinden minimum hata ile ev fiyatlarını tahmin 
eden bir makine öğrenmesi projesi gerçekleştirmek."""

# GEREKLİ KÜTÜPHANELER VE AYARLAR

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from helpers.data_prep import *
from helpers.eda import *

# Veri setinin yüklenmesi
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
print(train.shape)
print(test.shape)

###########################################
# 1. EXPLORATORY DATA ANALYSIS
############################################

#############################################
# Önemli Nümerik Değişkenlerin İncelenmesi
#############################################

# Nümerik değişkenlerin seçilmesi
num_cols = [col for col in train.columns if train[col].dtype in ['int64', 'float64']]

# Id & SalePrice değişkenlerinin çıkarılması
num_cols.remove('Id')
num_cols.remove('SalePrice')
num_analysis = train[num_cols].copy()

# Eksik gözlemlerin doldurulması
for col in num_cols:
    if num_analysis[col].isnull().sum() > 0:
        num_analysis[col] = SimpleImputer(strategy='median').fit_transform(num_analysis[col].values.reshape(-1, 1))

# Model
clf = ExtraTreesRegressor(random_state=42)
etreg_model = clf.fit(num_analysis, train.SalePrice)

# Önemli nümerik değişkenlerin görselleştirilmesi
def plot_importance(model, features, num=len(num_cols), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features})
    plt.figure(figsize=(16, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(etreg_model, num_cols)

# Bu değişkenlerin korelasyon matrisi
plt.figure(figsize=(8,8))
plt.title('Önemli değişkenlerin korelasyon matrisi')
cols =['OverallQual', 'GarageCars', 'GrLivArea', 'YearBuilt',
       'FullBath', '1stFlrSF', 'TotalBsmtSF', 'GarageArea','Fireplaces','GarageYrBlt','SalePrice']
sns.heatmap(train[cols].corr(),annot=True,square=True);

##### YORUM:
# GarageCars ile GarageArea arasında 0.88 lik pozitif yönlü korelasyon var yani ilişkileri yüksek.
# YearBuilt ile GarageYrBlt arasında 0.83 lük korelasyon var.
# 1stFlrSF ile TotalBsmtSF arasında 0.82 lik koralsyon var.

# https://www.kaggle.com/mviola/house-prices-eda-lasso-lightgbm-0-11635
def plot_numerical(col, discrete=False):
    if discrete:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.stripplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.countplot(train[col], ax=ax[1])
        fig.suptitle(str(col) + ' analysis')
    else:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.scatterplot(x=col, y='SalePrice', data=train, ax=ax[0])
        sns.distplot(train[col], kde=False, ax=ax[1])
        fig.suptitle(str(col) + ' analysis')


print('Plot functions are ready to use')

plot_numerical('OverallQual',discrete=True)
#### YORUM:
# Genel malzeme kalitesi yüksek olanların fiyatları daha yüksek.
# Ama 10. kalitede olup fiyatı düşük olan iki gözlem var bunlar aykırı olabilir.
# 5.ve 6. kalitede olan evlerin sayısı daha fazla.

plot_numerical('GarageCars',discrete=True)
### YORUM:
# Araç kapasitesi fazla olanların ev fiyatlarının yüksek olmasını bekleriz.
# Ama 3 araç kapasiteli garajı olan evlerin fiyatları 4 araç kapasiteli olanlardan daha fazla olması dikkat çekiyor.
# 4 araç kapasiteli garajı olan evlerin fiyatı daha düşük görünüyor.
# 2 araç kapasiteli garajı olan evlerin sayısı daha fazla.

plot_numerical('GrLivArea')
### YORUM:
# Metrekaresi yüksek olduğu halde(4000-5000) fiyatı düşük olan iki gözlem var,aykırı olabilir.
# GrLivArea ile SalePrice arasında doğrusal ilişki var gibi görünüyor.
# Evlerin oturma alanı genelde 1000 ile 2500 metrekare arasında yoğunluk göstermektedir.

plot_numerical('YearBuilt')
### YORUM:
# Yapım yılı 1880 ile 1900 arasında olup fiyatı yüksek olan evler var.Bunlar tarihi yapılar olabilir.
# 2000 yılı sonrasında yapılan evlerin fiyatları artmış.

# FullBath: Üst katlardaki tam banyolar
plot_numerical('FullBath',discrete=True)
### YORUM:
# Üst katta hiç banyosu olmadığı halde fiyatı yüksek olan evler var.

# 1stFlrSF : 1.kat metrekare alanı
plot_numerical('1stFlrSF')
### YORUM:
# 1.kat metrekare alanı büyük olup fiyatı düşük olan evler var.Diğer değişkenlerle incelenmeli.

# TotalBsmtSF : Kare ayaklı duvar kaplama alanı
plot_numerical('TotalBsmtSF')
### YORUM:
# TotalBsmtSF değişkeni toplam "Basement" yüzölçümünü veriyor."GrLivArea" ve "1stFlrSF"
# ve "2ndFlrSF" alan ölçümlerinin toplamı olduğu için bu değişkenlerin grafikleri
# ile benzer olduğu görülüyor.

plot_numerical('GarageArea')
# Garaj alanı büyük olup fiyatı düşük olan evler var.
# Hiç garajı olmayan evler var.

plot_numerical('Fireplaces',discrete=True)
### YORUM:
# 3 şöminesi olup fiyatı daha düşük olan evler var.

#################################################
# Önemli Kategorik Değişkenlerin İncelenmesi
#################################################

# Kategorik değişkenlerin seçilmesi
cat_features = [col for col in train.columns if train[col].dtype =='object']
cat_analysis = train[cat_features].copy()

for col in cat_analysis:
    if cat_analysis[col].isnull().sum() > 0:
        cat_analysis[col] = SimpleImputer(strategy='constant').fit_transform(cat_analysis[col].values.reshape(-1,1))

# One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
cat_analysis = one_hot_encoder(cat_analysis,cat_features)

# Model
clf = ExtraTreesRegressor(random_state=42)
h = clf.fit(cat_analysis, train.SalePrice)

# Önemli kategorik değişkenlerin görselleştirilmesi

def plot_importance(model, features, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(16, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:20])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(h, cat_analysis)

cat_analysis["SalePrice"] = train["SalePrice"]


def cat_plot(col1, col2):
    # tüm veri
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.stripplot(x=col1, y='SalePrice', data=train, ax=ax[0])
    sns.boxplot(x=col1, y='SalePrice', data=train, ax=ax[1])
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
    fig.suptitle(str(col1) + ' analysis')

    # one-hot encoding
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    sns.stripplot(x=col2, y='SalePrice', data=cat_analysis, ax=ax[0])
    sns.boxplot(x=col2, y='SalePrice', data=cat_analysis, ax=ax[1])
    fig.suptitle(str(col2) + ' analysis')

# ExterQual: Dış malzeme kalitesi
# Ex: Excellent
# Gd: Good
# TA: Average / Typical
# Fa: Fair
# Po: Poor
cat_plot("ExterQual","ExterQual_TA")

# BsmtQual: Bodrum yüksekliği
# Ex: Excellent (100+inches)
# Gd: Good (90-99)
# TA: Average / Typical (80-89)
# Fa: Fair (70-79)
# Po: Poor (<70)
# NA: No basement
cat_plot("BsmtQual","BsmtQual_Ex")

# FirePlaceQu: Şömine kalitesi
# Ex: Excellent-Exceptional masonry fireplace(olağanüstü duvar şöminesi)
# Gd: Good - masonry fireplace in main level(ana seviyede)
# TA: Average / Typical- Prefabricated Fireplace in main living area or Masonry Fireplace in basement(Prefabrik şömine ya da bodurm katında şömine)
# Fa: Fair- Prefabricated Fireplace in basement(bodrum prefabrik şömine)
# Po: Poor- Ben Franklin Stove
# NA: No fireplace
cat_plot("FireplaceQu","FireplaceQu_missing_value")

# Neighborhood: Ames şehir sınırları içindeki fiziksel konumları
# Neighborhood_NoRidge: Northridge
cat_plot("Neighborhood","Neighborhood_NoRidge")

# Diğer şehirlere bakıldığında en yüksek fiyatlı evler North Ridge de görünüyor bu nedenle
# SalePrice için belirleyici.Burada aykırılık olabilir.Aykırılıktan etkilenmiş olabilir.
# North Ridge de fiyatların yüksek olmasının sebebi California State Üniversitesinin bu ilçede
# olması olabilir,yani belki de aykırılık değildir.

####################################################
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
####################################################

#####################
# DATA PREPROCESSING
#####################

# MISSING VALUES

# Train ve test datalerını birleştirme
df = pd.concat([train, test]).reset_index(drop=True)
print(df.shape)

missing_values_table()

# Bazı değişkenledeki NAN olanları eksiklik olarak görüyor ama aslında bunlar
# eksiklik değil yokluğu ifade ediyor.Bu nedenle onları None olarak değiştireceğim.

none_cols = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
             'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']

# Sayısal değerlerdeki Nan olanlar aslında yokluk anlamında olduğu için onları da 0
# ile değiştireceğim.

zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
             'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']

# Diğer değişkenlerde eksiklik az olduğu için mod ile dolduracağım.
freq_cols = ['Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual',
             'SaleType', 'Utilities']

for col in zero_cols:
    df[col].replace(np.nan, 0, inplace=True)

for col in none_cols:
    df[col].replace(np.nan, 'None', inplace=True)

for col in freq_cols:
    df[col].replace(np.nan, df[col].mode()[0], inplace=True)

missing_values_table(df)

# MsZoning(genel imar sınıflandırması) değişkenindeki boş değerleri
# MSSubClassa(inşaat sınıfı) göre dolduracağım.

df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].apply(
    lambda x: x.fillna(x.mode()[0]))

# LotFrontage mülkiyetin cadde ile bağlantısını gösteren bir değişken, her mahallenin
# cadde bağlantısının birbirine benzeyebileceğinden bunu Neighborhood'a göre dolduracağım.

df['LotFrontage'] = df.groupby(
    ['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

missing_values_table(df)

# Sayısal değişken olup aslında kategorik değişken olması gerekenleri düzeltme
df['MSSubClass'] = df['MSSubClass'].astype(str)
df['YrSold'] = df['YrSold'].astype(str)
df['MoSold'] = df['MoSold'].astype(str)
df.info()

############################
# Feature Engineering
############################

# - LotShape(İmar şekli)
df["LotShape"].value_counts()

# LotShape(İmar şekli) değişkeninin IR2 ve IR3 alt sınıflarının frekansı düşük.
# Bu nedenle bunları IR1 e atayacağım.
df.loc[(df["LotShape"] == "IR2"), "LotShape"] = "IR1"
df.loc[(df["LotShape"] == "IR3"), "LotShape"] = "IR1"
df["LotShape"].value_counts()

# - ExterQual(Dış malzeme kalitesi)
df["ExterQual"].value_counts()

# Sınıflar arasında dengesizlik olduğu için Ex ve Gd(en iyi kaliteli iki sınıf) birleştirdik.
# TA ve Fa (orta kaliteli sınıf) birleştirdik
df.loc[df["ExterQual"]=="Ex","ExterQual"]=2
df.loc[df["ExterQual"]=="Gd","ExterQual"]=2
df.loc[df["ExterQual"]=="TA","ExterQual"]=1
df.loc[df["ExterQual"]=="Fa","ExterQual"]=1
df["ExterQual"]= df["ExterQual"].astype("int")
df["ExterQual"].value_counts()

# - BsmtQual(Bodrum Yüksekliği)
df["BsmtQual"].value_counts()

# Sınıflar arasında dengesizlik olduğu için Ex ve Gd(90 ile 100 inches arası) birleştirdik.
# TA ve Fa (70 ile 89 inches arası) birleştirdik.
# None da yokluk olduğu için dokunmadık.
df.loc[df["BsmtQual"]=="Ex","BsmtQual"]=2
df.loc[df["BsmtQual"]=="Gd","BsmtQual"]=2
df.loc[df["BsmtQual"]=="TA","BsmtQual"]=1
df.loc[df["BsmtQual"]=="Fa","BsmtQual"]=1
df.loc[df["BsmtQual"]=="None","BsmtQual"]=0
df["BsmtQual"]= df["BsmtQual"].astype("int")
df["BsmtQual"].value_counts()

# - KitchenQual(Mutfak kalitesi)
df["KitchenQual"].value_counts()

df.loc[df["KitchenQual"]=="Ex","KitchenQual"]=2
df.loc[df["KitchenQual"]=="Gd","KitchenQual"]=2
df.loc[df["KitchenQual"]=="TA","KitchenQual"]=1
df.loc[df["KitchenQual"]=="Fa","KitchenQual"]=1
df["KitchenQual"]= df["KitchenQual"].astype("int")
df["KitchenQual"].value_counts()

# - Neighborhood (Ames şehir sınırları içindeki fiziksel konum)
df.groupby("Neighborhood").agg({"SalePrice":"mean"}).sort_values(by="SalePrice", ascending=False)

# https://www.kaggle.com/oguzerdo/top-1-house-pricing-project-regression-models
# Target değişkeninin ortalamasına göre birbirine benzeyen ilçeleri birleştirdik.
neigh_map = {'MeadowV': 1,'IDOTRR': 1,'BrDale': 1,'BrkSide': 2,'OldTown': 2,'Edwards': 2,
             'Sawyer': 3,'Blueste': 3,'SWISU': 3,'NPkVill': 3,'NAmes': 3,'Mitchel': 4,
             'SawyerW': 5,'NWAmes': 5,'Gilbert': 5,'Blmngtn': 5,'CollgCr': 5,
             'ClearCr': 6,'Crawfor': 6,'Veenker': 7,'Somerst': 7,'Timber': 8,
             'StoneBr': 9,'NridgHt': 10,'NoRidge': 10}

df['Neighborhood'] = df['Neighborhood'].map(neigh_map).astype('int')


# Derecelendirme içeren değişkenleri ordinal yapıya getireceğim.
ext_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['ExterCond'] = df['ExterCond'].map(ext_map).astype('int')

bsm_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

df['BsmtCond'] = df['BsmtCond'].map(bsm_map).astype('int')

bsmf_map = {'None': 0,'Unf': 1,'LwQ': 2,'Rec': 3,'BLQ': 4,'ALQ': 5,'GLQ': 6}
df['BsmtFinType1'] = df['BsmtFinType1'].map(bsmf_map).astype('int')
df['BsmtFinType2'] = df['BsmtFinType2'].map(bsmf_map).astype('int')

heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['HeatingQC'] = df['HeatingQC'].map(heat_map).astype('int')

df['FireplaceQu'] = df['FireplaceQu'].map(bsm_map).astype('int')
df['GarageCond'] = df['GarageCond'].map(bsm_map).astype('int')
df['GarageQual'] = df['GarageQual'].map(bsm_map).astype('int')

rare_analyser(df, "SalePrice", 0.01)

# - LotConfig(Parsel yapılandırma)
df["LotConfig"].value_counts()

df.loc[(df["LotConfig"]=="Inside"),"LotConfig"] = 1
df.loc[(df["LotConfig"]=="FR2"),"LotConfig"] = 1
df.loc[(df["LotConfig"]=="Corner"),"LotConfig"] = 1

df.loc[(df["LotConfig"]=="FR3"),"LotConfig"] = 2
df.loc[(df["LotConfig"]=="CulDSac"),"LotConfig"] = 2
df["LotConfig"].value_counts()

# - LandSlope(Mülkün eğimi)
df["LandSlope"].value_counts()
df.loc[df["LandSlope"] == "Gtl", "LandSlope"] = 1
df.loc[df["LandSlope"] == "Sev", "LandSlope"] = 2
df.loc[df["LandSlope"] == "Mod", "LandSlope"] = 2
df["LandSlope"]= df["LandSlope"].astype("int")
df["LandSlope"].value_counts()

# - OverallQual(Genel malzeme ve bitiş kalitesi)
df["OverallQual"].value_counts()
df.loc[df["OverallQual"] == 1, "OverallQual"] = 1
df.loc[df["OverallQual"] == 2, "OverallQual"] = 1
df.loc[df["OverallQual"] == 3, "OverallQual"] = 1
df.loc[df["OverallQual"] == 4, "OverallQual"] = 2
df.loc[df["OverallQual"] == 5, "OverallQual"] = 3
df.loc[df["OverallQual"] == 6, "OverallQual"] = 4
df.loc[df["OverallQual"] == 7, "OverallQual"] = 5
df.loc[df["OverallQual"] == 8, "OverallQual"] = 6
df.loc[df["OverallQual"] == 9, "OverallQual"] = 7
df.loc[df["OverallQual"] == 10, "OverallQual"] = 8
df["OverallQual"].value_counts()

# - MasVnrType (Duvar kaplama türü)
df["MasVnrType"].value_counts()
df.loc[df["MasVnrType"] == "BrkCmn" , "MasVnrType"] = "None"
df["MasVnrType"].value_counts()

# - Foundation (Vakıf tipi)
df["Foundation"].value_counts()
df.loc[df["Foundation"] == "Stone", "Foundation"] = "BrkTil"
df.loc[df["Foundation"] == "Wood", "Foundation"] = "CBlock"
df["Foundation"].value_counts()

# - Fence (Çit kalitesi)
df["Fence"].value_counts()
df.loc[df["Fence"] == "MnWw", "Fence"] = "MnPrv"
df.loc[df["Fence"] == "GdWo", "Fence"] = "MnPrv"
df["Fence"].value_counts()

#################
# NEW FEATURES
#################

# Total banyo sayısı
df["TotalBath_NEW"] = df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5 + df['FullBath'] + df['HalfBath'] * 0.5

# Toplam Kat Sayısı
df['TotalSF_NEW'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])

# Toplam Veranda Alanı
df['TotalPorchSF_NEW'] = (df['OpenPorchSF'] + df['3SsnPorch'] +df['EnclosedPorch'] +df['ScreenPorch'] + df['WoodDeckSF'])

# Kaliteleriyle İlgili Değişkenler
df["OVER_QUAL_NEW"] = df['OverallQual'] + df['OverallCond']

# Bodrum Kalitesi
df["BSMT_QUAL_NEW"] = df['BsmtQual'] + df['BsmtCond']

# Dış Malzeme Kalitesi
df["EX_QUAL_NEW"] = df['ExterQual'] + df['ExterCond']

# Garaj Kalitesi
df['TotalGrgQual_NEW'] = (df['GarageQual'] + df['GarageCond'])

# Genel Kalite
df['TotalQual_NEW'] = df['OverallQual'] + df['EX_QUAL_NEW']  + df['TotalGrgQual_NEW'] + df['KitchenQual'] + df['HeatingQC']

# Lux Evler
df.loc[(df['Fireplaces'] > 0) & (df['GarageCars'] >= 3), "LUX_NEW"] = 1
df["LUX_NEW"].fillna(0, inplace=True)
df["LUX_NEW"] = df["LUX_NEW"].astype(int)

# Restore Edilmemiş-Edilmemiş
df.loc[df["YearBuilt"] == df["YearRemodAdd"], "NEW_home"] = 0
df.loc[df["YearBuilt"] != df["YearRemodAdd"], "NEW_home"] = 1

df['QualPorch_NEW'] = df['EX_QUAL_NEW'] * df['TotalPorchSF_NEW']

df['HasPool_NEW'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['Has2ndFloor_NEW'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['HasGarage_NEW'] = df['TotalGrgQual_NEW'].apply(lambda x: 1 if x > 0 else 0)
df['HasFireplace_NEW'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df['HasPorch_NEW'] = df['QualPorch_NEW'].apply(lambda x: 1 if x > 0 else 0)

# Bahçe Alanı
df["Garden_NEW"]=df["LotArea"] - df["GrLivArea"]

#################
# 3. OUTLIERS
#################

# outlier var mı?
for col in num_cols:
    print(col, check_outlier(df, col))
# outliers baskılanması
num_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
for col in num_cols:
    replace_with_thresholds(df, col)
# kontrol
for col in num_cols:
    print(col, check_outlier(df, col))

#####################
# RARE ENCODING
#####################

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "SalePrice", cat_cols)
df = rare_encoder(df, 0.01, cat_cols)
rare_analyser(df, "SalePrice", cat_cols)

# Billgi taşımayan değişkenler
useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.02).any(axis=None))]
useless_cols

cat_cols = [col for col in cat_cols if col not in useless_cols]
df.shape

# Bilgi taşımayan değişkenlerin çıkarılması
for col in useless_cols:
    df.drop(col, axis=1, inplace=True)
df.shape

rare_analyser(df, "SalePrice", cat_cols)

#######################
# One-Hot Encoding
#######################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Encoding den sonra bilgi taşımayan değişkenler oluştu mu?
rare_analyser(df, "SalePrice", cat_cols)

# Bilgi taşımayan değişkenler
useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]
useless_cols_new

# Bilgi taşımayan değişkenlerin çıkarılması
for col in useless_cols_new:
    df.drop(col, axis=1, inplace=True)
df.shape

######################
# 4. MODEL
######################
#test train setini önce birleştirmiştik şimdi ayıralım
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

##################
# Base Model
##################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

###############################
# Hyperparameter Optimization
###############################

lgbm_model = LGBMRegressor(random_state=46)

# Modelleme öncesi hata:
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model,
                                        X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

lgbm_params = {"learning_rate": [0.001, 0.01, 0.05, 0.1],
               "n_estimators": [200, 500, 750],
               "max_depth": [-1, 2, 5, 8],
               "colsample_bytree": [1, 0.50, 0.75],
               "num_leaves": [25, 31, 44]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

lgbm_gs_best.best_params_

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# Hiperparametrelerin default kendi değeriyle rmse 0.12676 idi.
# optimizasyonlarla 0.1228 e indirdik

#########################
# Feature Selection
#########################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(final_model, X, 50)

feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': X.columns})
feature_imp.sort_values(by="Value",ascending=False)

feature_imp[feature_imp["Value"] > 0].shape
feature_imp[feature_imp["Value"] < 1].shape

# Önemli olmayan değişkenlerin listesi
zero_imp_cols = feature_imp[feature_imp["Value"] < 1]["Feature"].values
zero_imp_cols

# Önemli olan değişkenler
selected_cols = [col for col in X.columns if col not in zero_imp_cols]
len(selected_cols)

###########################################################
# Hyperparameter Optimization with Selected Features
###########################################################

lgbm_model = LGBMRegressor(random_state=46)

lgbm_params = {"learning_rate": [0.001, 0.01, 0.05, 0.1],
               "n_estimators": [200, 500, 750],
               "max_depth": [-1, 2, 5, 8],
               "colsample_bytree": [1, 0.50, 0.75],
               "num_leaves": [25, 31, 44]}


lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=True).fit(X[selected_cols], y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X[selected_cols], y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X[selected_cols], y, cv=5, scoring="neg_mean_squared_error")))
rmse
#bir önceli rmse değeri 0.12281029384862836
#çöp değişkenlerle azaltmış olduk biraz da olsa 0.12315210453826239

##################################
# Sonuçların Yüklenmesi
##################################

submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"]
y_pred_sub = final_model.predict(test_df[selected_cols])

y_pred_sub = np.expm1(y_pred_sub) #ölçeklendirmiştik ya onu geri aldık

submission_df['SalePrice'] = y_pred_sub
submission_df["Id"]=submission_df["Id"].astype("int32")
submission_df.dtypes

submission_df.to_csv('submission.csv', index=False)
submission_df