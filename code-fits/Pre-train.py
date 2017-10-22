# Copyright (C) 2017 Zhixian MA <zx@mazhixian.me>
# Pretrain

import os
import time

def main():
    t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
    print("[%s] Pretrain on the augmented unLRG samples" % (t))

    # Train for 2 class classification
    t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
    print("[%s] Pretrain for 2 class..." % (t))
    os.system("python3 cae-Pretrain-2-class.py")

    # Train for 3 class classification
    t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
    print("[%s] Pretrain for 3 class..." % (t))
    os.system("python3 cae-Pretrain-3-class.py")

if __name__ == "__main__":
    main()



