


JOSA_LIST = ['한테', '의', '로써', '로서', '으로써', '에게서', '를', '으로서', '가', '까지', '이', '으로', '에게', '로', '와', '을', '부터', '에서', '과', '은', '는']
SINGULAR2PLURAL = {'를': '들을', '으로써': '들로써', '으로서': '들로서', '가': '들이', '게': '들에게', '으로': '들로', '와': '들과'}  # 단수 -> 복수
JOSA_PARALLEL = {'로': '으로', '를': '을', '가': '이',
                 '와': '과', '는': '은', '라': '이라', '랑': '이랑',
                 '로구': '이로구', '로군': '이로군', '며': '이며', '야': '이야',
                 '여': '이여', '다': '이다', '에요': '이에요', '예요': '이에요'}  # 받침 없을 때의 조사 -> 받침 있을 때의 조사
REVERSED_JOSA_PARALLEL = {val: key for key, val in JOSA_PARALLEL.items()}  # JOSA_PARALLEL의 반대 버전
REVERSED_JOSA_PARALLEL['이에요'] = '예요'

with open('assets/popular_nouns.txt', 'r', encoding='utf8') as f:
    POPULAR_NOUNS = f.read().split('\n')
with open('assets/pronouns.txt', 'r', encoding='utf8') as f:
    PRONOUNS = f.read().split('\n')
with open('assets/replace.txt', 'r', encoding='utf8') as f:
    REPLACES = {i.split('\t')[0] : i.split('\t')[1] for i in f.read().split('\n')}
