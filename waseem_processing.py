from encodings import utf_8
import re

f = open("waseemDataSet.txt", encoding="utf8")
count = 0
racism = []
sexism = []
none = []
#print(words[len(words)-1] + "aaaaa")
for line in f:
    count += 1
    line = re.sub(r'[^a-zA-Z\n]', ' ', line)
    line = re.sub(' +', ' ', line)
    check_line = "Line{}: {}".format(count, line.strip())
    if check_line[len(check_line)-6:] == "racism":
        racism.append(line + "\n")
    if check_line[len(check_line)-6:] == "sexism":
        sexism.append(line + "\n")
    if check_line[len(check_line)-4:] == "none":
        none.append(line + "\n")

f.close()

test = open("testset.txt", "w", encoding='utf8')
dev = open("devset.txt", "w", encoding='utf8')
train = open("trainset.txt", "w", encoding='utf8')

train.write(''.join(sexism[0:int(len(sexism)*.8)]) + ''.join(racism[0:int(len(racism)*.8)]) + ''.join(none[0:int(len(none)*.8)]))
dev.write(''.join(sexism[int(len(sexism)*.8):int(len(sexism)*.9)]) + ''.join(racism[int(len(racism)*.8):int(len(racism)*.9)]) + ''.join(none[int(len(none)*.8):int(len(none)*.9)]))
test.write(''.join(sexism[int(len(sexism)*.9):]) + ''.join(racism[int(len(racism)*.9):]) + ''.join(none[int(len(none)*.9):]))

test.close()
dev.close()
train.close()