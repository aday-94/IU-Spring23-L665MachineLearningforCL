feature_matrix = []
file1 = open("testset.txt", 'r')
lexicon1 = open("hate_lexicon_wiegand.txt", 'r')
features = open("features.txt", "w")

file = file1.readlines()
lexicon = lexicon1.readlines()

for word in lexicon:
    feature_vector = []
    for line in file:
        if word.strip() in line:
            feature_vector.append(1)
        else:
            feature_vector.append(0)
    feature_matrix.append(feature_vector)

#print(feature_matrix)

for feature_vector in feature_matrix:
    features.writelines(str(feature_vector) + '\n')

file1.close()
lexicon1.close()
features.close()