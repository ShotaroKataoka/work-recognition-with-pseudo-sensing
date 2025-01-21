import logging
import os

def get_logger(log_dir, name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ファイルハンドラー
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
