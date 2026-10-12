import json

INPUT  = "tutorial.ipynb"
OUTPUT = "tutorial.ipynb"

with open(INPUT, "r", encoding="utf-8") as f:
    nb = json.load(f)

def code_cell(src_str):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [src_str],
        "execution_count": None,
        "outputs": []
    }

cells = nb["cells"]

# Cell 21 - use ALL features
cells[21] = code_cell(
    "# Use ALL features (we will handle categoricals below)\n"
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "# Reload original CSVs so we have all columns\n"
    "current_dir = Path(os.getcwd())\n"
    'train_file = current_dir / "input" / "train.csv"\n'
    'test_file  = current_dir / "input" / "test.csv"\n'
    'sample_sub_file = current_dir / "input" / "sample_submission.csv"\n'
    "\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import warnings\n"
    'warnings.filterwarnings("ignore")\n'
    "\n"
    "train_full = pd.read_csv(train_file)\n"
    "test_full  = pd.read_csv(test_file)\n"
    "sample_sub = pd.read_csv(sample_sub_file)\n"
    "\n"
    'target = train_full["TARGET"].values\n'
    "\n"
    "# Drop ID and target from feature sets\n"
    'drop_cols = ["SK_ID_CURR", "TARGET"]\n'
    "train_feats = train_full.drop(columns=drop_cols)\n"
    'test_feats  = test_full.drop(columns=["SK_ID_CURR"])\n'
    "\n"
    'print("Train features shape:", train_feats.shape)\n'
    'print("Test  features shape:", test_feats.shape)\n'
)

# Cell 29 - check missing values train
cells[29] = code_cell(
    "# Check missing values of train data (top 20 columns)\n"
    "miss = train_feats.isnull().sum()\n"
    "miss = miss[miss > 0].sort_values(ascending=False)\n"
    'print("Columns with missing values:", len(miss))\n'
    "miss.head(20)\n"
)

# Cell 30 - check missing values test
cells[30] = code_cell(
    "# Check missing values of test data (top 20)\n"
    "miss_test = test_feats.isnull().sum()\n"
    "miss_test = miss_test[miss_test > 0].sort_values(ascending=False)\n"
    'print("Test columns with missing values:", len(miss_test))\n'
    "miss_test.head(20)\n"
)

# Cell 68 - full preprocessing with LabelEncoder
cells[68] = code_cell(
    "# Full preprocessing: encode categoricals, leave NaNs\n"
    "# for LightGBM (it handles missing values natively)\n"
    "from sklearn.preprocessing import LabelEncoder\n"
    "\n"
    "train_X = train_feats.copy()\n"
    "test_X  = test_feats.copy()\n"
    "\n"
    "cat_cols = train_X.select_dtypes(include=['object']).columns.tolist()\n"
    'print("Categorical columns ({}): {}".format(len(cat_cols), cat_cols))\n'
    "\n"
    "le = LabelEncoder()\n"
    "for col in cat_cols:\n"
    "    combined = pd.concat([train_X[col], test_X[col]], axis=0).astype(str)\n"
    "    le.fit(combined)\n"
    "    train_X[col] = le.transform(train_X[col].astype(str))\n"
    "    test_X[col]  = le.transform(test_X[col].astype(str))\n"
    "\n"
    'print("Encoding done.")\n'
    'print("train_X shape:", train_X.shape)\n'
)

# Cell 70 - placeholder
cells[70] = code_cell(
    "# Categorical encoding already done above.\n"
    'print("All categoricals encoded via LabelEncoder.")\n'
)

# Cell 72 - feature engineering
cells[72] = code_cell(
    "# Feature Engineering\n"
    "for df in [train_X, test_X]:\n"
    '    df["CREDIT_INCOME_RATIO"]   = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)\n'
    '    df["ANNUITY_INCOME_RATIO"]  = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)\n'
    '    df["CREDIT_GOODS_RATIO"]    = df["AMT_CREDIT"] / (df["AMT_GOODS_PRICE"] + 1)\n'
    '    ext_cols = [c for c in df.columns if c.startswith("EXT_SOURCE")]\n'
    "    if ext_cols:\n"
    '        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)\n'
    '        df["EXT_SOURCE_STD"]  = df[ext_cols].std(axis=1)\n'
    '        df["EXT_SOURCE_MIN"]  = df[ext_cols].min(axis=1)\n'
    '    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)\n'
    '    df["DAYS_EMPLOYED"]      = df["DAYS_EMPLOYED"].replace(365243, np.nan)\n'
    "\n"
    'print("train_X shape after FE:", train_X.shape)\n'
)

# Cell 74 - OWN_CAR_AGE clip
cells[74] = code_cell(
    "for df in [train_X, test_X]:\n"
    '    if "OWN_CAR_AGE" in df.columns:\n'
    '        df.loc[df["OWN_CAR_AGE"] >= 60, "OWN_CAR_AGE"] = np.nan\n'
    'print("OWN_CAR_AGE clipped.")\n'
)

# Cell 76 - placeholder
cells[76] = code_cell('print("OWN_CAR_AGE left as-is for LightGBM.")\n')

# Cell 77 - placeholder
cells[77] = code_cell(
    'print("Final feature count:", train_X.shape[1])\n'
    "print(train_X.dtypes.value_counts())\n"
)

# Cell 81 - imports
cells[81] = code_cell(
    "import lightgbm as lgb\n"
    "from sklearn.model_selection import StratifiedKFold\n"
    "from sklearn.metrics import roc_auc_score\n"
    "import numpy as np\n"
    'print("LightGBM version:", lgb.__version__)\n'
)

# Cell 83 - align + prepare arrays
cells[83] = code_cell(
    'train_X, test_X = train_X.align(test_X, join="left", axis=1)\n'
    "\n"
    "X       = train_X.values\n"
    "y       = target\n"
    "X_test  = test_X.values\n"
    "feature_names = list(train_X.columns)\n"
    "\n"
    'print("X shape       :", X.shape)\n'
    'print("X_test shape  :", X_test.shape)\n'
    'print("y distribution:", np.bincount(y))\n'
)

# Cell 85 - no scaling needed
cells[85] = code_cell(
    "# LightGBM does NOT require feature scaling.\n"
    'print("No scaling needed for LightGBM.")\n'
)

# Cell 87 - K-fold setup
cells[87] = code_cell(
    "N_FOLDS  = 5\n"
    "SEED     = 42\n"
    "skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)\n"
    "\n"
    "oof_preds  = np.zeros(len(X))\n"
    "test_preds = np.zeros(len(X_test))\n"
    "\n"
    'print("Starting {}-fold CV ...".format(N_FOLDS))\n'
)

# Cell 89 - LightGBM training loop
cells[89] = code_cell(
    "lgb_params = {\n"
    '    "objective"        : "binary",\n'
    '    "metric"           : "auc",\n'
    '    "boosting_type"    : "gbdt",\n'
    '    "n_estimators"     : 2000,\n'
    '    "learning_rate"    : 0.02,\n'
    '    "num_leaves"       : 63,\n'
    '    "max_depth"        : -1,\n'
    '    "min_child_samples": 80,\n'
    '    "subsample"        : 0.8,\n'
    '    "subsample_freq"   : 1,\n'
    '    "colsample_bytree" : 0.6,\n'
    '    "reg_alpha"        : 0.1,\n'
    '    "reg_lambda"       : 0.1,\n'
    '    "scale_pos_weight" : float((y == 0).sum()) / float((y == 1).sum()),\n'
    '    "random_state"     : SEED,\n'
    '    "n_jobs"           : -1,\n'
    '    "verbose"          : -1,\n'
    "}\n"
    "\n"
    "fold_scores = []\n"
    "\n"
    "for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):\n"
    "    X_tr, X_val = X[tr_idx], X[val_idx]\n"
    "    y_tr, y_val = y[tr_idx], y[val_idx]\n"
    "\n"
    "    model = lgb.LGBMClassifier(**lgb_params)\n"
    "    model.fit(\n"
    "        X_tr, y_tr,\n"
    "        eval_set=[(X_val, y_val)],\n"
    "        callbacks=[\n"
    "            lgb.early_stopping(stopping_rounds=100, verbose=False),\n"
    "            lgb.log_evaluation(period=200)\n"
    "        ]\n"
    "    )\n"
    "\n"
    "    val_pred  = model.predict_proba(X_val)[:, 1]\n"
    "    oof_preds[val_idx] = val_pred\n"
    "    test_preds += model.predict_proba(X_test)[:, 1] / N_FOLDS\n"
    "\n"
    "    score = roc_auc_score(y_val, val_pred)\n"
    "    fold_scores.append(score)\n"
    '    print("  Fold {} AUC: {:.5f}  |  best iter: {}".format(fold, score, model.best_iteration_))\n'
    "\n"
    "oof_score = roc_auc_score(y, oof_preds)\n"
    'print("\\nOOF AUC : {:.5f}".format(oof_score))\n'
    'print("Mean CV AUC: {:.5f}".format(np.mean(fold_scores)))\n'
)

# Cell 90 - feature importance plot
cells[90] = code_cell(
    "import matplotlib.pyplot as plt\n"
    "\n"
    "fi = pd.DataFrame({\n"
    '    "feature"   : feature_names,\n'
    '    "importance": model.feature_importances_\n'
    '}).sort_values("importance", ascending=False)\n'
    "\n"
    "plt.figure(figsize=(10, 8))\n"
    'plt.barh(fi["feature"].head(30)[::-1], fi["importance"].head(30)[::-1])\n'
    'plt.xlabel("LightGBM feature importance (gain)")\n'
    'plt.title("Top 30 features")\n'
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "fi.head(20)\n"
)

# Cell 93 - summary
cells[93] = code_cell(
    'print("OOF AUC     : {:.5f}".format(oof_score))\n'
    "for i, s in enumerate(fold_scores, 1):\n"
    '    print("  Fold {}: {:.5f}".format(i, s))\n'
    'print("Mean CV AUC : {:.5f}".format(np.mean(fold_scores)))\n'
)

# Cell 98 - final prediction
cells[98] = code_cell(
    "pred = test_preds\n"
    'print("Test prediction stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(pred.min(), pred.max(), pred.mean()))\n'
)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Done! tutorial.ipynb has been updated.")
