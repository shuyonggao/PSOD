#！/bin/bash

# python train_1st_round.py   # step1

# python test_DUTS.py   # step2    generate DUTS_train maps

# python utils/denseCRF.py    # step3    finetune DUTS_train maps
  
# python utils/NSS_2nd_GtMask.py     # step4 

python train_2nd_round.py  # step5

python test.py  # step6  test step, you should set the path of the test images
