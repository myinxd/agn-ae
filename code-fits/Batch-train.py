# Copyright (C) 2017 Zhixian MA <zx@mazhixian.me>

import os
import time

def main():
    numepoch = 9
    # timestamp
    t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
    print("[%s] Number of training and testing circles are %d" % (t, numepoch))

    for i in range(0,numepoch):
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training and testing circle: %d" % (t, i))

        subfold = "./results_171021_pre/subfold{0}".format(i)
        # Randomly generate training samples
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Randomly generating samples for training..." % (t))
        # os.system("python3 genLRGAug-fits.py")
        # Train layer1
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training the first layer..." % (t))
        # os.system("python3 cnn-Compact-vs-Mrph.py %s" % (subfold))
        # Train layer2
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training the second layer..." % (t))
        # os.system("python3 cnn-FR-vs-A.py %s" % (subfold))
        # Train layer3
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training the third layer..." % (t))
        # os.system("python3 cnn-FRI-vs-FRII.py %s" % (subfold))
         # Train layer4
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training the fourth layer..." % (t))
        # os.system("python3 cnn-BT-vs-Ir.py %s" % (subfold))
        # Train layer5
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training the fifth layer..." % (t))
        os.system("python3 cnn-X-vs-R.py %s" % (subfold))
        # Estimation layer1
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Estimating the first layer..." % (t))
        os.system("python3 Estimation-labeled-l1.py %s" % (subfold))
        # Estimation layer2
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Estimating the second layer..." % (t))
        os.system("python3 Estimation-labeled-l2.py %s" % (subfold))
        # Estimation layer3
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Estimating the third layer..." % (t))
        os.system("python3 Estimation-labeled-l3.py %s" % (subfold))
        # Estimation layer4
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Estimating the fourth layer..." % (t))
        os.system("python3 Estimation-labeled-l4.py %s" % (subfold))
        # Estimation layer5
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Estimating the fifth layer..." % (t))
        os.system("python3 Estimation-labeled-l5.py %s" % (subfold))
        # Final coding
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Final coding..." % (t))
        os.system("python3 Get_final_code_labeled.py %s %d" % (subfold, i))

if __name__ == "__main__":
    main()
