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
                    param_tmp = '-'+key+' '+self.optdict[key]
                else:
                    param_tmp = key+' '+self.optdict[key]
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
    parser = argparse.ArgumentParser(description="Transform fits to other form.")
    # Parameters
    parser.add_argument("foldname", help="Path of the fits samples.")
    parser.add_argument("savefolder", help="The folder to save transformed files.")
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

    # transform
    optdict = {
        "cmap": "Heat",
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
        path_out = os.path.join(savefolder,img_name)
        # open fits and save image
        print("Processing on sample %s" % s[0:-5])
        savecmd = "-saveimage jpeg " + path_out + " 100"
        finalcmd =" ".join([opt.gen_cmd(filepath=path_in),savecmd, '&'])
        try:
            os.system(finalcmd)
        except:
            continue
        time.sleep(1)
        # kill
        f = os.popen('ps -ef | grep ds9')
        text = f.readline()
        pid = text.split(" ")[6]
        os.kill(int(pid), signal.SIGKILL)

if __name__ == "__main__":
    main()
