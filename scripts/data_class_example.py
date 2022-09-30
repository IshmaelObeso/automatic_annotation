import pandas as pd
from components import prediction_data_pipe

dataset = r"C:\Users\gobes\Documents\Unannoted_Exports\spectral_triplets\2022-09-29,20-08-31.129178"
data_class = prediction_data_pipe.Data_Pipe(dataset)

all_uids = data_class.get_uids()
example_uid = all_uids[0]

# test gettitem
xs, ys, ts, uid = data_class.__getitem__(example_uid)

# test getitem for every uid
for uid in all_uids:

    xs, ys, ts, uid = data_class.__getitem__(uid)


    print(uid)