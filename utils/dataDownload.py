# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
This script assist us to fetch observations, i.e., the sample images from
the archive. In this work, the data are from the FIRST archive.

Target URL: https://third.ucllnl.org/cgi-bin/firstcutout/

References
==========
[1] python3-requests documentation
    https://docs.python-requests.org/en/master/user/quickstart/
[2] How to avoid the "SSL: CERTIFICATE_VERIFY_FAILED"
    https://blog.csdn.net/xiaopangxia/article/details/49908889/
[3] Python-requests introduction
    https://blog.csdn.net/shanzhizi/article/details/50903748/
"""

# import ssl
import os
import requests
import numpy as np
import argparse

class DataFetcher():
    """
    A class to fetch or download data from the url.

    Inputs
    ======
    url: str
        The url of the archive or dataset
    params: dict
        The dict hosts required parameters, e.g., "Name": name

    Methods
    =======
    get_params_update: update the params
    get_url: get the constructed url
    get_reponse: get data with 'requests.get' method
    save_data: save the data w.r.t. to required filetype
    """

    def __init__(self, url=None, params=None):
        """The initializer"""
        self.url = url
        self.params = params
        self.data_requests = None

    def get_params(self):
        """Get parameter"""
        return self.params

    def get_url(self):
        """Get the url"""
        return self.url

    def get_params_update(self, params_update):
        """Update the parameter dict

        Inputs
        ======
        params_update: dict
            The params to be updated

        """
        if isinstance(params_update, dict)==False:
            print("The updated parameters should be set as dict.")
            return None
        # update
        for key in params_update.keys():
            if key in self.params.keys():
                self.params[key] = params_update[key]
            else:
                print("Parameter %s does not exist." % key)
                continue

    def get_response(self):
        """Fetch data from the website.
        """
        try:
            self.data_requests = requests.get(url=self.url,
                                              params=self.params)
        except: #ssl.SSLError:
            self.data_requests = requests.get(url=self.url,
                                              params=self.params,
                                              verify=False)

    def save_data(self,savepath):
        """Save the data to provided path

        savepath: str
            The file path to save fetched data.
        """
        # if self.data_requests is None:
        if not os.path.exists(savepath):
            self.get_response()
            # raw response content
            filesfx = savepath.split('.')[-1]
            if filesfx != 'fits':
                print("Warning: file type %s may not be opened correctly." % filesfx)
            with open(savepath, 'wb') as fd:
                for chunk in self.data_requests.iter_content(chunk_size=128):
                    fd.write(chunk)

def bin2csv(binpath, savepath):
    """Readjust the binary file to readable file."""
    # Open files
    fb = open(binpath, 'rb')
    fs = open(savepath, 'w')

    # The first line
    for line in  fb.readlines():
        bin_items = str(line,'utf-8').split(' ')
        bin_items = np.array(bin_items)
        bin_items = np.delete(bin_items, np.where(bin_items==''))
        new_line = ' '.join(bin_items)
        # write
        fs.write('%s' % new_line)

    fb.close()
    fs.close()

def batch_download(dataFetcher, listpath, batch, savefolder):
    """Batch download the samples

    Inputs
    ======
    dataFetecher: class DataFetcher
        The data fetcher that hosts url and params
    listpath: str
        The path of the data list
    batch: tuple
        The region of indices w.r.t. samples to be fetched.
    savefolder: str
        Folder to save the fetched sample files
    """
    import time
    timestamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    print('[%s]: Downloading samples from %s' % (timestamp, dataFetcher.get_url()))

    # open listpath
    f = open(listpath,'rb')
    lines = f.readlines() # The first line is not required.
    # regularize the batch
    if batch[0] <=0:
        batch[0] = 1
    if batch[1] > len(lines):
        batch[1] = len(lines)
    # log file optional
    fl = open('log.txt', 'a')
    # Iteration body
    for i in range(batch[0], batch[1]+1):
        # timestamp
        t = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        # get params
        sample_line = str(lines[i], 'utf-8').split(' ')
        update_ra = ' '.join(sample_line[0:6])
        update_param = {'RA': update_ra}
        # update param
        dataFetcher.get_params_update(params_update=update_param)
        # download file
        fname = 'J' + ''.join(sample_line[0:6]) + '.fits'
        savepath = os.path.join(savefolder,fname)
        try:
            dataFetcher.save_data(savepath)
        except:
            fl.write("%d: %s" % (i, fname))
            continue
        # print log
        print('[%s]: Fetching %s' % (t, fname))
    f.close()
    fl.close()

def batch_download_csv(dataFetcher, listpath, batch, savefolder):
    """Batch download the samples

    Inputs
    ======
    dataFetecher: class DataFetcher
        The data fetcher that hosts url and params
    listpath: str
        The path of the data list
    batch: tuple
        The region of indices w.r.t. samples to be fetched.
    savefolder: str
        Folder to save the fetched sample files
    """
    from pandas import read_csv
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import time
    timestamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    print('[%s]: Downloading samples from %s' % (timestamp, dataFetcher.get_url()))

    # load csv
    f = read_csv(listpath, sep=' ')
    ra = f['RAJ2000'] # RA
    dec = f['DEJ2000'] # DEC
    # regularize the batch
    if batch[1] > len(f):
        batch[1] = len(f)
    # log file optional
    fl = open('log.txt', 'a')
    # Iteration body
    for i in range(batch[0], batch[1]+1):
        # timestamp
        t = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        # get params
        temp_c = SkyCoord(ra=ra[i]*u.degree, dec=dec[i]*u.degree, frame='icrs')
        # Coordinate transform
        ra_rms = tuple(temp_c.ra.hms)
        dec_dms = tuple(temp_c.dec.dms)
        ra_h = str(int(ra_rms[0]))
        ra_m = str(int(ra_rms[1]))
        ra_s = str(np.round(ra_rms[2]*1000)/1000)
        de_d = str((dec_dms[0]))
        de_m = str(int(np.abs(dec_dms[1])))
        de_s = str(np.abs(np.round(dec_dms[2]*1000)/1000))
        update_ra = ' '.join([ra_h,ra_m,ra_s,de_d,de_m,de_s])
        update_param = {'RA': update_ra}
        # update param
        dataFetcher.get_params_update(params_update=update_param)
        # download file
        fname = 'J' + ''.join([ra_h,ra_m,ra_s,de_d,de_m,de_s]) + '.fits'
        savepath = os.path.join(savefolder,fname)
        try:
            dataFetcher.save_data(savepath)
        except:
            fl.write("%d: %s" % (i, fname))
            continue
        # print log
        print('[%s]: Fetching %s' % (t, fname))
    fl.close()

def batch_download_excel(dataFetcher, listpath, batch, savefolder):
    """Batch download the samples

    Inputs
    ======
    dataFetecher: class DataFetcher
        The data fetcher that hosts url and params
    listpath: str
        The path of the data list
    batch: tuple
        The region of indices w.r.t. samples to be fetched.
    savefolder: str
        Folder to save the fetched sample files
    """
    from pandas import read_excel
    import time
    timestamp = time.strftime('%Y-%m-%d: %H:%M:%S',time.localtime(time.time()))
    print('[%s]: Downloading samples from %s' % (timestamp, dataFetcher.get_url()))

    # load csv
    f = read_excel(listpath)
    samples = f.get_values()
    ra = samples[:,1] # RA
    name = samples[:,1] # Sample name
    # regularize the batch
    if batch[1] > len(samples):
        batch[1] = len(samples)
    # log file optional
    fl = open('log.txt', 'a')
    # Iteration body
    for i in range(batch[0], batch[1]):
        # timestamp
        t = time.strftime('%Y-%m-%d: %H:%M:%S',time.localtime(time.time()))
        # get params
        RA_h = ra[i][2:4]
        RA_m = ra[i][4:6]
        RA_s = ra[i][6:11]
        DEC_d = ra[i][11:14]
        DEC_a = ra[i][14:16]
        DEC_s = ra[i][16:20]
        if DEC_d[0] == '−':
            DEC_d = '-' + DEC_d[1:]
        update_param = {'RA': " ".join([RA_h, RA_m, RA_s, DEC_d, DEC_a, DEC_s])}
        print(update_param['RA'])
        # update param
        dataFetcher.get_params_update(params_update=update_param)
        # download file
        fname_temp = name[i].split(" ")
        fname = "_".join(fname_temp[1:-2]) + '.fits'
        savepath = os.path.join(savefolder,fname)
        print("[%s] processing on sample fname %s" % (t, fname))
        try:
            dataFetcher.save_data(savepath)
        except:
            fl.write("%d: %s" % (i, fname))
            continue
        # print log
        # print('[%s]: Fetching %s' % (t, fname))
    fl.close()

def batch_download_txt(dataFetcher, listpath, batch, savefolder):
    """Batch download the samples

    Inputs
    ======
    dataFetecher: class DataFetcher
        The data fetcher that hosts url and params
    listpath: str
        The path of the data list
    batch: tuple
        The region of indices w.r.t. samples to be fetched.
    savefolder: str
        Folder to save the fetched sample files
    """
    import time
    timestamp = time.strftime('%Y-%m-%d: %H:%M:%S',time.localtime(time.time()))
    print('[%s]: Downloading samples from %s' % (timestamp, dataFetcher.get_url()))

    # load csv
    with open(listpath, 'r') as fp:
        samples = fp.readlines()

    # regularize the batch
    if batch[1] > len(samples):
        batch[1] = len(samples)
    # log file optional
    fl = open('log.txt', 'a')
    # Iteration body
    for i in range(batch[0], batch[1]):
        # timestamp
        t = time.strftime('%Y-%m-%d: %H:%M:%S',time.localtime(time.time()))
        s = samples[i].split(" ")
        ra = s[1]
        # get params
        RA_h = ra[1:3]
        RA_m = ra[3:5]
        RA_s = ra[5:10]
        DEC_d = ra[10:13]
        DEC_a = ra[13:15]
        DEC_s = ra[15:19]
        if DEC_d[0] == '−':
            DEC_d = '-' + DEC_d[1:]
        update_param = {'RA': " ".join([RA_h, RA_m, RA_s, DEC_d, DEC_a, DEC_s])}
        print(update_param['RA'])
        # update param
        dataFetcher.get_params_update(params_update=update_param)
        # download file
        fname_temp = s[1]
        fname = fname_temp + '.fits'
        savepath = os.path.join(savefolder,fname)
        print("[%s] processing on sample fname %s" % (t, fname))
        try:
            dataFetcher.save_data(savepath)
        except:
            fl.write("%d: %s" % (i, fname))
            continue
        # print log
        # print('[%s]: Fetching %s' % (t, fname))
    fl.close()

def batch_download_dat(dataFetcher, listpath, batch, savefolder):
    """Batch download the samples

    Inputs
    ======
    dataFetecher: class DataFetcher
        The data fetcher that hosts url and params
    listpath: str
        The path of the data list
    batch: tuple
        The region of indices w.r.t. samples to be fetched.
    savefolder: str
        Folder to save the fetched sample files
    """
    import time
    timestamp = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    print('[%s]: Downloading samples from %s' % (timestamp, dataFetcher.get_url()))

    # open listpath
    with open(listpath,'rb') as f:
        lines = f.readlines() # The first line is not required.
    # regularize the batch
    if batch[0] < 0:
        batch[0] = 0
    if batch[1] > len(lines):
        batch[1] = len(lines)
    # log file optional
    fl = open('log.txt', 'a')
    # Iteration body
    for i in range(batch[0], batch[1]):
        # timestamp
        t = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        # get params
        sample_line = str(lines[i], 'utf-8').split(' ')
        RA_h = sample_line[4]
        RA_m = sample_line[5]
        RA_s = sample_line[6]
        DEC_d = sample_line[8]
        DEC_m = sample_line[9]
        DEC_s = sample_line[10]
        update_ra = " ".join([RA_h, RA_m, RA_s, DEC_d, DEC_m, DEC_s])
        update_param = {'RA': update_ra}
        # update param
        dataFetcher.get_params_update(params_update=update_param)
        # download file
        RA_h = "%02d" % int(sample_line[4])
        RA_m = "%02d" % int(sample_line[5])
        RA_s = float(sample_line[6])
        RA_s_i = np.fix(np.round(RA_s*100)/100)
        RA_s_f = np.round(RA_s*100)/100 - RA_s_i
        RA_s_new = "%02d.%02d" % (int(RA_s_i), int(RA_s_f*100))
        DEC_d = sample_line[8]
        if DEC_d[0] == '+':
            DEC_d = "+%02d" % (int(DEC_d[1:]))
        else:
            DEC_d = "-%02d" % (int(DEC_d[1:]))
        DEC_m = "%02d" % int(sample_line[9])
        DEC_s = float(sample_line[10])
        DEC_s_i = np.fix(np.round(DEC_s*10)/10)
        DEC_s_f = np.round(DEC_s*10)/10 - DEC_s_i
        DEC_s_new = "%02d.%01d" % (int(DEC_s_i), int(DEC_s_f*10))
        fname = 'J' + ''.join([RA_h, RA_m, RA_s_new, DEC_d, DEC_m, DEC_s_new]) + ".fits"
        savepath = os.path.join(savefolder,fname)
        try:
            dataFetcher.save_data(savepath)
        except:
            fl.write("%d: %s" % (i, fname))
            continue
        # print log
        print('[%s]: Fetching %s' % (t, fname), i)
    f.close()
    fl.close()

def main():
    # Init
    parser = argparse.ArgumentParser(description="Fetch FIRST observations.")
    # Parameters
    # parser.add_argument("url", help="URL of the archive'")
    parser.add_argument("listpath", help="Path of the sample list.")
    parser.add_argument("batchlow", help="Begin index of the batch.")
    parser.add_argument("batchhigh",help="End index of the batch.")
    parser.add_argument("savefolder", help="The folder to save files.")
    args = parser.parse_args()

    listpath = args.listpath
    batch = [int(args.batchlow),int(args.batchhigh)]
    savefolder = args.savefolder

    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    # Init DataFetcher
    url = "https://third.ucllnl.org/cgi-bin/firstcutout"
    params = {
        'RA': '10 50 07.270 +30 40 37.52',
        'Dec': '',
        'Equinox': 'J2000',
        'ImageSize': '4.5',
        'ImageType': 'FITS Image',
        'MaxInt': 200,
        '.submit': 'Extract the Cutout',
    }
    df = DataFetcher(url=url, params=params)

    # download
    batch_download_dat(dataFetcher=df,
                   listpath=listpath,
                   batch=batch,
                   savefolder=savefolder)

if __name__ == "__main__":
    main()
