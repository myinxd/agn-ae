# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Transform the fits image to jpeg form after some preprocessings.

Reference
=========
ds9 command line options: http://ds9.si.edu/doc/ref/command.html
"""

import os
import time
import signal
import argparse


class ParamDS9():
    """
    A class to generate ds9 command line options

    example
    =======
    >>> optdict = {
                "cmap": "Heat",
                "smooth": "radius 4",
                "zoom": "to 3",
                }
    >>> opt = ParamDS9(optdict=optdict)
    >>> cmd_line = opt.gen_cmd(filepath="ds9.fits")
    """

    def __init__(self, optdict=None):
        self.optdict = optdict
        self.gen_opt()

    def gen_opt(self):
        """Generate the ds9 command line options"""

        # Init
        self.optlist = []
        if isinstance(self.optdict, dict):
            for key in self.optdict:
                if key[0] != '-':
                    param_tmp = '-' + key + ' ' + self.optdict[key]
                else:
                    param_tmp = key + ' ' + self.optdict[key]
                # append
                self.optlist.append(param_tmp)
        else:
            print("Option dictionary shoud be provided.")
            return

        self.optcmd = " ".join(self.optlist)

    def gen_cmd(self, filepath=None):
        """Generate the final command line sentence"""
        cmdline = " ".join(["ds9", filepath, self.optcmd])

        return cmdline


def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(
        description="Transform fits to other form.")
    # Parameters
    parser.add_argument("foldname", help="Path of the fits samples.")
    parser.add_argument(
        "savefolder", help="The folder to save transformed files.")
    parser.add_argument("batchlow")
    parser.add_argument("batchhigh")
    args = parser.parse_args()

    foldname = args.foldname
    savefolder = args.savefolder
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    # Load sample list
    try:
        samplelist = os.listdir(foldname)
    except:
        print("The folder %s does not exist." % foldname)
        return

    batchlow = int(args.batchlow)
    batchhigh = int(args.batchhigh)
    if batchhigh > len(samplelist):
        batchhigh = len(samplelist)
    samplelist = samplelist[batchlow:batchhigh]
    # transform
    optdict = {
        "cmap": "a",
        "smooth": "radius 4",
        "zoom": "to 3",
        "-width": "400",
        "height": "400",
    }
    # ParamDS9
    opt = ParamDS9(optdict=optdict)

    for s in samplelist:
        path_in = os.path.join(foldname, s)
        img_name = s[0:-5] + '.jpeg'
        path_out = os.path.join(savefolder, img_name)
        print("Processing on sample %s" % s[0:-5])
        if os.path.exists(path_out):
            continue
        else:
            # open fits and save image
            savecmd = "-saveimage jpeg " + path_out + " 100"
            finalcmd = " ".join([opt.gen_cmd(filepath=path_in), savecmd, '&'])
            try:
                os.system(finalcmd)
            except:
                continue
                print('Wrong')
            # dog watch
            flag = 0
            while flag == 0:
                if os.path.exists(path_out):
                    flag = 1
                    # kill the process
                    f = os.popen('ps -ef | grep ds9')
                    text = f.readline().split(" ")
                    flag_kill = 1
                    while flag_kill != 0:
                        if text[flag_kill] != "":
                            pid = text[flag_kill]
                            flag_kill = 0
                        else:
                            flag_kill = flag_kill + 1
                    try:
                        print(pid)
                        os.kill(int(pid), signal.SIGKILL)
                    except:
                        print("Something wrong with the pid")
            # time.sleep(2)
                else:
                    time.sleep(2)
                    print("Wong~Wong~")

if __name__ == "__main__":
    main()
