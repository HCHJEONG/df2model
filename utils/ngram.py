class Ngram():
    def __init__(self, a, b) -> None:
        # a = "오늘 강남에서 맛있는 스파게티를 먹었다."
        # b = "강남에서 먹었던 오늘의 스파게티는 맛있었다."
        # 2-gram
        r2, word2 = self.diff_ngram(a, b, 2)
        print("2-gram:", r2, word2)
        # 3-gram
        r3, word3  = self.diff_ngram(a, b, 3)
        print("3-gram:", r3, word3)

    def ngram(self, s, num):
        res = []
        slen = len(s) - num + 1
        for i in range(slen):
            ss = s[i:i+num]
            res.append(ss)
        return res

    def diff_ngram(self, sa, sb, num):
        a = self.ngram(sa, num)
        b = self.ngram(sb, num)
        r = []
        cnt = 0
        for i in a:
            for j in b:
                if i == j:
                    cnt += 1
                    r.append(i)
        return cnt / len(a), r
    