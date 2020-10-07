from zipfile import ZipFile 
import pandas as pd
import io
import os
import re

def load_lob(path):
    """Reads all .zip files in a given path and returns Pandas data frame"""
    # a list of .zip files in
    zip_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.zip')]
    bytes_buffer = io.BytesIO()

    for f in zip_files:
        print(f'Reading {f}...')
        with ZipFile(f) as zf:
            for z in zf.filelist:
                print('Writing to buffer...')
                bytes_buffer.write(zf.read(z))

    bytes_buffer.seek(0)
    print('')
    print('Writing to data frame...')
    lob = pd.read_fwf(bytes_buffer, encoding='utf-8', index_col=0, header=None)
    bytes_buffer.close()
    return(lob)

if __name__ == '__main__':
    ZIPDIR = '/Users/Logan/Desktop/Capstone'
    load_lob(ZIPDIR)
