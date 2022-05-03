from saliency_toolbox import calculate_measures
import numpy as np
import matplotlib.pyplot as plt

## plot F-measure curve and Precision-Recall curve

prec      = np.load('save/Precision.npy')
recall    = np.load('save/Recall.npy')
f_measure = np.load('save/Fmeasure_all_thresholds.npy')
plt.plot(recall, prec)
plt.figure()
plt.plot(np.linspace(0, 1, len(f_measure)), f_measure)
plt.show()


# rs_dirs = ['AFNet','BASNet','DGRL','EGNet','F3Net','GateNet','LDF' ,'GCPANet','MINet','MLMS','Ours','PiCANet-R','PoolNet','R3+','RAS']
# lineSylClr=['g-',    'y-',   'b-',   'm-',   'c-',    'k-',  'b-.'    ,  'r--',   'g--', 'y--',  'r-',   'b--',    'm--',   'c--', 'k--' ]
# linewidth = [1,1,1, 1,1,1, 1,1,1, 1,2,1,1, 1,1]
# ##------ add by us---------------##
# plot_save_pr_curves(PRE, # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
#                     REC, # numpy array (num_rs_dir,255)
#                     method_names = rs_dirs, # method names, shape (num_rs_dir), will be included in the figure legend
#                     lineSylClr = lineSylClr, # curve styles, shape (num_rs_dir)
#                     linewidth = linewidth, # curve width, shape (num_rs_dir)
#                     xrange = (0,1.0), # the showing range of x-axis  (0.5,1.0)
#                     yrange = (0,1.0), # the showing range of y-axis   (0.5,1.0)
#                     dataset_name = target_dataset, # dataset name will be drawn on the bottom center position
#                     save_dir = save_pic_dir, # figure save directory
#                     save_fmt = 'png') # format of the to-be-saved figure
# print('\n')
#
# ## 4. =======Plot and save F-measure curves=========
# print("------ 4. Plot and save F-measure curves------")
# plot_save_fm_curves(FM, # numpy array (num_rs_dir,255), num_rs_dir curves will be drawn
#                     mybins = np.arange(0,256),
#                     method_names = rs_dirs, # method names, shape (num_rs_dir), will be included in the figure legend
#                     lineSylClr = lineSylClr, # curve styles, shape (num_rs_dir)
#                     linewidth = linewidth, # curve width, shape (num_rs_dir)
#                     xrange = (0.0,1.0), # the showing range of x-axis
#                     yrange = (0.0,1.0), # the showing range of y-axis
#                     dataset_name = target_dataset, # dataset name will be drawn on the bottom center position
#                     save_dir = save_pic_dir, # figure save directory
#                     save_fmt = 'png') # format of the to-be-saved figure
# print('\n')
#
# print('Done!!!')