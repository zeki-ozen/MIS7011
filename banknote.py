import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

current_date = datetime.datetime.now().strftime("%d%m%Y_%H%M")
banknote_authentication = fetch_ucirepo(id=267)
# data (as pandas dataframes)
X = banknote_authentication.data.features
y = banknote_authentication.data.targets.iloc[:, 0]