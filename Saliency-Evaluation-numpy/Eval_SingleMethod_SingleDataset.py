import glob
from saliency_toolbox import calculate_measures
import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook

# 将文件写入excel表格和txt中
record = 'SgMeth_SgDataset'
# xlsx_name = record+'.xlsx'  # 测试完一种方法记录一次
logfile = record+'.txt' # 每测试完一个数据集记录一次


method = 'wssa'
datasetname = 'scribble 50 epoch 10500img vgg16'

gt_dir = '/home/gaosy/DATA/GT/PASCAL_S/mask'  # 'GT/'
sm_dir = '/home/gaosy/WSOD_compute_prfm/WSSA/PASCAL_S'  # 'SM/'

res=calculate_measures(gt_dir, sm_dir, ['MAE', 'Adp-E-measure', 'S-measure', 'Max-F', 'Mean-F', 'Adp-F', 'Wgt-F'],
                         save=False) # 'MAE', 'Adp-E-measure', 'S-measure', 'Max-F', 'Mean-F', 'Adp-F', 'Wgt-F'

with open(logfile, 'a') as f:  # 'a' 打开文件接着写
    f.write("\n------------cut off line--------------\n")
    f.write('{} dataset with {} method get {:.4f} mae, {:.4f} adp-e-measure, '
            '{:.4f} s-measure, {:.4f} adp-f, {:.4f} wgt-f, {:.4f} max-f, {:.4f} mean-f \n'.format(
        datasetname, method, res['MAE'], res['Adp-E-measure'],
        res['S-measure'], res['Adp-F'], res['Wgt-F'],
        res['Max-F'], res['Mean-F']))

print('{} dataset with {} method get {:.4f} mae, {:.4f} adp-e-measure, '
            '{:.4f} s-measure, {:.4f} adp-f, {:.4f} wgt-f, {:.4f} max-f, {:.4f} mean-f \n'.format(
        datasetname, method, res['MAE'], res['Adp-E-measure'],
        res['S-measure'], res['Adp-F'], res['Wgt-F'],
        res['Max-F'], res['Mean-F']))

# # 评测完一种方法保存一次
# # 如果没有excel 创建一个，并保存头
# if not os.path.exists(xlsx_name):
#     writer = pd.ExcelWriter(xlsx_name)  # 关键2，创建名称为hhh的excel表格
#     DTset = np.array(
#         ['', 'DUTS_test', '', '', '', 'DUT_O', '', '', '', 'ECSSD', '', '', '', 'PASCAL_S', '', '', '', 'HKU_IS',
#          '', '', '', 'SOD', '', '']).reshape(1, -1)
#     EVLmetric = np.array(['maxF', 'meanF', 'S-m', 'MAE',
#                           'maxF', 'meanF', 'S-m', 'MAE',
#                           'maxF', 'meanF', 'S-m', 'MAE',
#                           'maxF', 'meanF', 'S-m', 'MAE',
#                           'maxF', 'meanF', 'S-m', 'MAE',
#                           'maxF', 'meanF', 'S-m', 'MAE', ]).reshape(1, -1)
#     head = np.r_[DTset, EVLmetric]
#     head = pd.DataFrame(head)
#     head.to_excel(writer, float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
#     writer.save()
#
# # 在原有数据的基础上增加新行
# data_sourece = pd.DataFrame(pd.read_excel(xlsx_name))  # 读取原数据文件和表 ,sheet_name='aa'
# book = load_workbook(xlsx_name)
# writer = pd.ExcelWriter(xlsx_name, engine='openpyxl') # 重新创建这个表格
# writer.book = book
# writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
# df_rows = data_sourece.shape[0]  # 获取原数据的行数
#
# data = [method]
# for dsn in ['DUTS_test', 'DUT_O', 'ECSSD', 'PASCAL_S', 'HKU_IS', 'SOD']:
#     for metric in ['Max-F', 'Mean-F', 'S-measure', 'MAE']:
#         data.append(format(res[dsn][metric], '.4f'))
# data = np.array(data).reshape(1,-1)
#
# DATA = pd.DataFrame(data)
# DATA.to_excel(writer, startrow=df_rows + 1,index=False, header=False)  # 将数据写入excel中的aa表,从第一个空行开始写
# writer.save()  # 保存