# coding: UTF-8
import sqlite3,pprint
mode = 1;       #モード指定  0:英和  1:和英
#ダウンロードした wnjpn.dbのパスを指定
conn = sqlite3.connect("./wnjpn.db")
cur = conn.cursor()
print("running")
if mode == 0:	#英和用SQL
    cur.execute( "\
        SELECT DISTINCT word_en.wordid, word_en.lemma,  word_ja.lemma\
        FROM sense sense_A, sense sense_B , word word_en , word word_ja \
        WHERE word_en.wordid = sense_A.wordid\
            and sense_A.lang = 'eng'\
            and sense_B.lang = 'jpn'\
            and sense_A.synset = sense_B.synset\
            and sense_B.wordid = word_ja.wordid\
        ORDER BY word_en.lemma"\
    )
else :				#和英用SQL
    cur.execute( "\
        SELECT DISTINCT word_ja.wordid, word_ja.lemma,  word_en.lemma\
        FROM sense sense_A, sense sense_B , word word_en , word word_ja \
        WHERE word_ja.wordid = sense_A.wordid\
            and sense_A.lang = 'jpn'\
            and sense_B.lang = 'eng'\
            and sense_A.synset = sense_B.synset\
            and sense_B.wordid = word_en.wordid\
        ORDER BY word_ja.lemma ASC"\
    )
#ワードIDを整理しつつ、結果を一度配列に詰める。
#英単語に含まれるアンダーバーはスペースであるべきなので置換する(ハイフンはハイフンのままで良い)。
before = -1
word_list = []
i = -1
for row in cur:
    if before != row[0]:
        i = i + 1
        word_list.append( [row[1].replace("_"," "),row[2].replace("_"," ")])
        before = row[0]
    else :
        word_list[i][1] += ',' + row[2].replace("_"," ")
#タブ区切りでファイルに出力
f = open('./ja_to_eng.tsv', 'w')
i = 1
for word in word_list:
    f.write(str(i) + "\t" + word[0] +"\t" + word[1] +"\n")
    i += 1
f.close()
print("done")
cur.close()
conn.close()