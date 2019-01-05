#encoding utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Toolkit for Writing System Modeling
# Copyright (c) 2018, Dengliang Shi & Pingping Chen
# Author: Dengliang Shi, dengliang.shi@yahoo.com
#         Pingping Chen, pingping.chen@gmail.com
# Apache 2.0 License
# --------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries
import os

# Third-party Libraries


# User Defined Modules
from model import Model
from utils import parser
from utils import set_logger

# ------------------------------------------------------------Global--------------------------------------------------------


# -------------------------------------------------------------Main---------------------------------------------------------
args = parser.parse_args()
log_file = os.path.join(args.output_path, 'lstmlm.log')
logger = set_logger('LSTMLM', log_file)

if __name__ == '__main__':
    model = Model()
    model.init_model(logger, args)
    model.train()