#encoding utf-8

# ---------------------------------------------------------Libraries--------------------------------------------------------
# Standard Libraries
import logging

# Third-party Libraries


# User Define Modules
from .args import parser

# --------------------------------------------------------Global Strings----------------------------------------------------


# -------------------------------------------------------------Main---------------------------------------------------------
def set_logger(name, log_file):
    """Set up a logger for recording information during processing.
    :Param name    : specify a name for logger.
    :Param log_file: path of file for saving logs.
    """
    # set up a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create a file handler and set level to debug
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # define a formatter for log message
    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(format_string)
    # set message format of file and console handler
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add file and console handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def print_args(title, args):
    """print out all given arguments.
    :Param title: title will be showed before arguments.
    :Param args : arguments need be showed.
    """
    print(title)
    for key, value in args.items():
        print('%s: %r' % (key, value))