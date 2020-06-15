class KFoldTargetEncoderTrain_std(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, colnames,targetName,n_fold=10,verbosity=True,discardOriginal_col=False, threshold= 1, showPrint= False):

        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col
        self.threshold= threshold
        self.showPrint= showPrint

    def fit(self, X, y=None):
        return self

    def transform(self,X):

        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)
        assert(type(self.threshold) == int)
       
        kf = KFold(n_splits = self.n_fold, shuffle = False, random_state=2019)
        #kf = StratifiedKFold(n_splits = self.n_fold, shuffle = False, random_state=2019)

        col_mean_name = self.colnames + '_' + 'enc_mean'
        col_std_name= self.colnames + '_' + 'enc_std'
        
        #Le inizializzo con tutti NaN
        X[col_mean_name] = np.nan
        X[col_std_name] = np.nan
        
        #A ogni giro (per ogni K-Fold) fillo valori della Test Fold con la media delle Train Fold.
        for tr_ind, val_ind in kf.split(X):
            if self.showPrint is True:
                print('Indici delle Train-Folds: {},\033[1mTest Fold da fillare: {}\033[0m \n'.format(tr_ind, val_ind))  #sono gli indici delle folds

            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]

            diz_mean= X_tr.groupby(self.colnames)[self.targetName].mean().to_dict()
            diz_std= X_tr.groupby(self.colnames)[self.targetName].std().to_dict()

            #### Tratto come Nan chi non supera la threshold, agli altri assegno media e std sulle altre folds
            frequenze= X_tr.groupby(self.colnames).size()
            cat_oltre_threshold= frequenze[frequenze >= self.threshold].index
            
            if self.showPrint == True:
                print('cat_oltre_threshold', cat_oltre_threshold)

            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].apply(lambda x: diz_mean[x] if x in cat_oltre_threshold else np.nan )
            X.loc[X.index[val_ind], col_std_name] = X_val[self.colnames].apply(lambda x: diz_std[x] if x in cat_oltre_threshold else np.nan)

            mean_of_target = X_tr[self.targetName].mean()  #media di tutto il target sul KFold Train, not grouped
            std_of_target= X_tr[self.targetName].std()    # # std di tutto il target sul KFold Train, not grouped

            X[col_mean_name].fillna(mean_of_target, inplace = True) #ai NaN metto la media del KFold Train Target
            X[col_std_name].fillna(std_of_target, inplace = True)   # ai NaN metto la std dev media del Kfold Train Target

        if self.verbosity:

            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                    self.targetName,np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
            
        if self.discardOriginal_col:
            if self.showPrint == True:
                print('columns: ', X.columns)
            X = X.drop(self.colnames, axis=1)
        return X
    
    
class KFoldTargetEncoderTest_std(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self,train,colNames,Train_encoded_mean, Train_encoded_std, showPrint= False):
        
        self.train = train
        self.colNames = colNames
        self.Train_encoded_mean = Train_encoded_mean
        self.Train_encoded_std = Train_encoded_std
        self.showPrint= showPrint
        
    def fit(self, X, y=None):
        return self

    def transform(self,X):

        ### Media e std del Target grouped per categoria su tutto il Train
        category_mean_tr = self.train[[self.colNames,self.Train_encoded_mean]].groupby(self.colNames).mean() #.reset_index() 
        category_std_tr= self.train[[self.colNames,self.Train_encoded_std]].groupby(self.colNames).mean() #.reset_index() 
        
        ### Media dell'Encoding basato su mean e std sul Train (non grouped per categoria)
        population_mean_tr= self.train[self.Train_encoded_mean].mean()
        population_std_tr= self.train[self.Train_encoded_std].mean()
        
        ### Metto nel Test i valori medi della mean e std del Train per quella categoria
        X[self.Train_encoded_mean]= X[self.colNames].map(category_mean_tr.squeeze().to_dict())
        X[self.Train_encoded_std]= X[self.colNames].map(category_std_tr.squeeze().to_dict())
        
        ### Fillo i Missing Values (Categorie Nuove) con i valori medi di mean e std (encodate) di tutte la variabile nel Train
        if self.showPrint== True:
            print('Filled {} Missing Values with Average mean and std equals to {}, {}'.format(
                                                        np.sum(np.sum(X.isna())), population_mean_tr , population_std_tr))
        
        X[self.Train_encoded_mean].fillna(population_mean_tr, inplace = True) #fillo con media di enc_mean sul Train
        X[self.Train_encoded_std].fillna(population_std_tr, inplace = True)   #fillo con media di enc_std sul Train
        
        return X