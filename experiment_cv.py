import os
import sys
import gc
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

current_dir = Path.cwd()
train_file = current_dir / 'input' / 'train.csv'
test_file  = current_dir / 'input' / 'test.csv'

train_full = pd.read_csv(train_file)
test_full  = pd.read_csv(test_file)
target = train_full['TARGET'].values

drop_cols = ['SK_ID_CURR', 'TARGET']
train_feats = train_full.drop(columns=drop_cols)
test_feats  = test_full.drop(columns=['SK_ID_CURR'])

train_X = train_feats.copy()
test_X  = test_feats.copy()

cat_cols = train_X.select_dtypes(include=['object']).columns.tolist()

for col in cat_cols:
    le = LabelEncoder()
    comb = pd.concat([train_X[col], test_X[col]], axis=0).astype(str)
    le.fit(comb)
    train_X[col] = le.transform(train_X[col].astype(str))
    test_X[col]  = le.transform(test_X[col].astype(str))

# Base tutorial features
for df in [train_X, test_X]:
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED_CLEAN'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1.0)
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1.0)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1.0)
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1.0)
    df['GOODS_CREDIT_DIFF'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    df['INCOME_CREDIT_DIFF'] = df['AMT_INCOME_TOTAL'] - df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1.0)
    df['CREDIT_PER_PERSON'] = df['AMT_CREDIT'] / (df['CNT_FAM_MEMBERS'] + 1.0)
    df['ANNUITY_PER_PERSON'] = df['AMT_ANNUITY'] / (df['CNT_FAM_MEMBERS'] + 1.0)

    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    df['EMPLOYED_YEARS'] = -df['DAYS_EMPLOYED_CLEAN'] / 365.25
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED_CLEAN'] / (df['DAYS_BIRTH'] - 1.0)
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / (df['DAYS_BIRTH'] - 1.0)
    df['REG_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / (df['DAYS_BIRTH'] - 1.0)
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_BIRTH'] - 1.0)
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (df['AGE_YEARS'] + 1.0)
    df['CAR_TO_EMPLOYED_RATIO'] = df['OWN_CAR_AGE'] / (df['EMPLOYED_YEARS'] + 1.0)

    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCES_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCES_STD']  = df[ext_cols].std(axis=1)
    df['EXT_SOURCES_MIN']  = df[ext_cols].min(axis=1)
    df['EXT_SOURCES_MAX']  = df[ext_cols].max(axis=1)
    df['EXT_SOURCES_PROD'] = df['EXT_SOURCE_1'].fillna(1) * df['EXT_SOURCE_2'].fillna(1) * df['EXT_SOURCE_3'].fillna(1)
    df['EXT_1_x_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_2_x_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_1_x_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCES_OVER_AGE'] = df['EXT_SOURCES_MEAN'] / (df['AGE_YEARS'] + 1.0)
    df['EXT_2_OVER_AGE'] = df['EXT_SOURCE_2'] / (df['AGE_YEARS'] + 1.0)
    df['EXT_3_OVER_AGE'] = df['EXT_SOURCE_3'] / (df['AGE_YEARS'] + 1.0)
    df['EXT_SOURCE_WEIGHTED'] = (
        df['EXT_SOURCE_1'].fillna(0.5) * 1.5 +
        df['EXT_SOURCE_2'].fillna(0.5) * 3.0 +
        df['EXT_SOURCE_3'].fillna(0.5) * 3.5
    )

    bureau_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]
    df['BUREAU_TOTAL_REQ'] = df[bureau_cols].sum(axis=1)
    df['BUREAU_MON_TO_YEAR'] = df['AMT_REQ_CREDIT_BUREAU_MON'] / (df['AMT_REQ_CREDIT_BUREAU_YEAR'] + 1.0)

for grp in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'NAME_EDUCATION_TYPE']:
    for col in ['EXT_SOURCE_2', 'AMT_CREDIT']:
        grp_mean = train_X.groupby(grp)[col].mean()
        train_X[f'{col}_BY_{grp}_MEAN'] = train_X[grp].map(grp_mean)
        test_X[f'{col}_BY_{grp}_MEAN']  = test_X[grp].map(grp_mean)
        train_X[f'{col}_BY_{grp}_RATIO'] = train_X[col] / (train_X[f'{col}_BY_{grp}_MEAN'] + 1e-5)
        test_X[f'{col}_BY_{grp}_RATIO']  = test_X[col] / (test_X[f'{col}_BY_{grp}_MEAN'] + 1e-5)

for df in [train_X, test_X]:
    if "OWN_CAR_AGE" in df.columns:
        df.loc[df["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = np.nan

# Confirmed beneficial additions:
comb_raw = pd.concat([train_full, test_full], axis=0).reset_index(drop=True)
for col in ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE']:
    freq = comb_raw[col].value_counts().to_dict()
    train_X[f'{col}_COUNT'] = train_full[col].map(freq)
    test_X[f'{col}_COUNT']  = test_full[col].map(freq)

for df in [train_X, test_X]:
    df['EXT_1_ISNA'] = df['EXT_SOURCE_1'].isna().astype(int)
    df['EXT_3_ISNA'] = df['EXT_SOURCE_3'].isna().astype(int)
    df['TOTAL_NAN_COUNT'] = df.isna().sum(axis=1)
    # Additional high-signal ratios:
    df['DISPOSABLE_INCOME'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']
    df['INCOME_TO_EMPLOYED'] = df['AMT_INCOME_TOTAL'] / (df['EMPLOYED_YEARS'] + 1.0)
    df['CREDIT_TO_EMPLOYED'] = df['AMT_CREDIT'] / (df['EMPLOYED_YEARS'] + 1.0)

print(f"Features: {train_X.shape[1]}")

# Setup cross-validation
N_FOLDS = 5
SEED = 42
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

te_cols = [
    col for col in [
        'ORGANIZATION_TYPE', 'OCCUPATION_TYPE',
        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE'
    ]
    if col in train_full.columns
]
te_train_categories = train_full[te_cols].fillna('__MISSING__').astype(str).reset_index(drop=True)

def smoothed_target_encoding(reference_categories, reference_target, apply_categories, smoothing=20.0):
    prior = float(np.mean(reference_target))
    stats = pd.DataFrame({
        'category': reference_categories.to_numpy(),
        'target': reference_target
    }).groupby('category')['target'].agg(['sum', 'count'])
    encoded = (stats['sum'] + smoothing * prior) / (stats['count'] + smoothing)
    return apply_categories.map(encoded).fillna(prior).to_numpy(dtype=float)

X = train_X.values
y = target

fold_indices = list(skf.split(X, y))
tr_idx, val_idx = fold_indices[0]

X_tr_base, X_val_base = X[tr_idx], X[val_idx]
y_tr, y_val = y[tr_idx], y[val_idx]
categories_tr = te_train_categories.iloc[tr_idx].reset_index(drop=True)
categories_val = te_train_categories.iloc[val_idx].reset_index(drop=True)

inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
te_tr = np.zeros((len(tr_idx), len(te_cols)), dtype=float)
for inner_tr_idx, inner_val_idx in inner_skf.split(X_tr_base, y_tr):
    for col_idx, col in enumerate(te_cols):
        te_tr[inner_val_idx, col_idx] = smoothed_target_encoding(
            categories_tr[col].iloc[inner_tr_idx], y_tr[inner_tr_idx],
            categories_tr[col].iloc[inner_val_idx]
        )

te_val = np.column_stack([
    smoothed_target_encoding(categories_tr[col], y_tr, categories_val[col])
    for col in te_cols
])

X_tr_model = np.column_stack([X_tr_base, te_tr])
X_val_model = np.column_stack([X_val_base, te_val])

# LGB with Seed 42
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 60,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.1,
    'reg_lambda': 10.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}
m_lgb1 = lgb.LGBMClassifier(**lgb_params)
m_lgb1.fit(X_tr_model, y_tr, eval_set=[(X_val_model, y_val)], callbacks=[lgb.early_stopping(80, verbose=False)])
p_lgb1 = m_lgb1.predict_proba(X_val_model)[:, 1]
print(f"LGB Fold 1 AUC: {roc_auc_score(y_val, p_lgb1):.5f}")

# XGBoost
xgb_params = {
    'n_estimators': 1400,
    'learning_rate': 0.03,
    'max_depth': 4,
    'min_child_weight': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'reg_lambda': 10.0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
}
m_xgb1 = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=80)
m_xgb1.fit(X_tr_model, y_tr, eval_set=[(X_val_model, y_val)], verbose=False)
p_xgb1 = m_xgb1.predict_proba(X_val_model)[:, 1]
print(f"XGB Fold 1 AUC: {roc_auc_score(y_val, p_xgb1):.5f}")

# CatBoost Native
cat_params = {
    'iterations': 1200,
    'learning_rate': 0.03,
    'depth': 6,
    'eval_metric': 'AUC',
    'l2_leaf_reg': 10.0,
    'random_seed': 42,
    'verbose': 0,
    'thread_count': -1,
}
cat_train_frame = train_X.copy().reset_index(drop=True)
for categorical_col in cat_cols:
    cat_train_frame[categorical_col] = train_feats[categorical_col].fillna('__MISSING__').astype(str).to_numpy()

m_cat = cb.CatBoostClassifier(**cat_params)
m_cat.fit(
    cat_train_frame.iloc[tr_idx], y_tr,
    cat_features=cat_cols,
    eval_set=(cat_train_frame.iloc[val_idx], y_val),
    early_stopping_rounds=60
)
p_cat = m_cat.predict_proba(cat_train_frame.iloc[val_idx])[:, 1]
print(f"Native CAT Fold 1 AUC: {roc_auc_score(y_val, p_cat):.5f}")

# Blend comparison:
p_blend = 0.3 * p_lgb1 + 0.2 * p_cat + 0.5 * p_xgb1
print(f"Blend Fold 1 AUC: {roc_auc_score(y_val, p_blend):.5f}")
