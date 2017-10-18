# Copyright (C) 2017 Zhixian MA <zx@mazhixian.me>

import os
import time

def main():
    numepoch = 10
    # timestamp
    t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
    print("[%s] Number of training and testing circles are %d" % (t, numepoch))

    for i in range(numepoch):
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training and testing circle: %d" % (t, i))

        subfold = "./results/subfold{0}".format(i)
        # Train layer1
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training the first layer..." % (t))
        os.system("python3 cnn-Compact-vs-Mrph.py %s" % (subfold))
        # Train layer2
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training the second layer..." % (t))
        os.system("python3 cnn-FRI-vs-FRII-vs-Other.py %s" % (subfold))
        # Train layer3
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training the third layer..." % (t))
        os.system("python3 cnn-BT-vs-X-vs-R.py %s" % (subfold))
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
        # Final coding
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Final coding..." % (t))
        os.system("python3 Get_final_code_labeled.py %s %d" % (subfold, i))

if __name__ == "__main__":
    main()



