from tqdm import tqdm

from ap.ap_logger import logging

class ProgressBarLogger:
    def __init__(self, total=None, desc=None, leave=True):
        self.pbar = tqdm(total=total, desc=desc, leave=leave)
        self.pbar.n = 0

    def update(self, n=1):
        self.pbar.update(n)
        percentage = int(self.pbar.n / self.pbar.total * 100)
        if percentage % 5 == 0:
            logging.info(self.pbar.format_dict)

    def close(self):
        self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
