import pandas as pd
import numpy as np
from ambrosia.splitter import Splitter

df_1 = pd.DataFrame(np.random.normal(size=(5_000, 2)), columns=["metric_val", "metric_val2"])
df_1["frame"] = 1

df_2 = pd.DataFrame(np.random.normal(size=(5_000, 2)), columns=["metric_val", "metric_val2"])
df_2["frame"] = 2

df = pd.concat([df_1, df_2])

# splitter = Splitter()
# factor = 0.5

# res = splitter.run(dataframe=df, method="hash", part_of_table=factor, salt="bug")

# print(res.group.value_counts())

from ambrosia.designer import Designer

designer = Designer()

res = designer.run(to_design="effect", effects=1.01, metrics="metric_val")

print(res)
