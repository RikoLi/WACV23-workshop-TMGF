# TMGF full model
# market
python train.py --conf configs/TMGF_full.yml TASK_NAME TMGF_training_market DATASET.ROOT_DIR ./datasets DATASET.NAME Market1501 MODEL.SIE_CAMERA 6 MEMORY_BANK.POS_K 3
# duke
# python train.py --conf configs/TMGF_full.yml TASK_NAME TMGF_training_duke DATASET.ROOT_DIR ./datasets DATASET.NAME DukeMTMC-reID MODEL.SIE_CAMERA 8 MEMORY_BANK.POS_K 2
# msmt
# python train.py --conf configs/TMGF_full.yml TASK_NAME TMGF_training_msmt DATASET.ROOT_DIR ./datasets DATASET.NAME MSMT17 MODEL.SIE_CAMERA 15 MEMORY_BANK.POS_K 3