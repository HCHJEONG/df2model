import codecs
from konlpy.tag import Twitter

def word_count(lines: list):
    
    twitter = Twitter()
    word_dic = {}
    for line in lines:
        malist = twitter.pos(line)
        for word in malist:
            if word[1] == "Noun": #  명사 확인하기 --- (※3)
                if not (word[0] in word_dic):
                    word_dic[word[0]] = 0
                word_dic[word[0]] += 1 # 카운트하기
    # 많이 사용된 명사 출력하기 --- (※4)
    keys = sorted(word_dic.items(), key=lambda x:x[1], reverse=True)
    for word, count in keys[:50]:
        print("{0}({1}) ".format(word, count), end="")
    print()

    return keys

print(Twitter().pos('소유권이전'))