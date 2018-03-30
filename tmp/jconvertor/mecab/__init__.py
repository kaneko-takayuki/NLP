# -*- coding: utf-8 -*-

import sys

from tmp.nattopy.natto import MeCab

sys.stdout.write("MeCabのモデルを読み取っています...")
sys.stdout.flush()
tagger = MeCab(r"-O '' -F%m\t%f[7]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n -U%m\t%m\t%m\t%F-[0,1,2,3]\t\t\n -d /var/lib/mecab/dic/unidic")
print("完了")
