# 以下脚本加载原始数据集，并将日期时间信息解析为Pandas Data Frame索引。
# “No”列被删除，然后为每列指定更清晰的名称。最后，将NA值替换为“0”值，并删除前24小时。
from pandas import read_csv
from datetime import datetime

# load data


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


dataset = read_csv('F:/情感计算/数据集/Beijing+PM2.5+Data/PRSA_data_2010.1.1-2014.12.31.csv',
                   parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)

# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)

# drop the first 24 hours
dataset = dataset[24:]

# summarize first 5 rows
print(dataset.head(5))

# save to file
dataset.to_csv('pollution.csv')