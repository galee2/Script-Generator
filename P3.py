##########################################################################################################
# Author: Gene Lee
# CS 540 (Summer 2021)
########################################################################################################## 

import numpy as np
import math
import string
import re

############################################# HELPER CLASSES #############################################


############################################ HELPER FUNCTIONS ############################################
def prob_calc(count, tot, laplace, smooth):
    if smooth == True: 
        p = round((count + 1) / (tot + laplace), 4)
        if (p < 0.0001): p = 0.0001
    else: 
        p = round(count / tot, 4)

    return p

def bigram_prob_fix(bigram):
    for i in range(27):
        p_tot = round(sum(bigram[i]), 4)
        diff = p_tot - 1
        done = -1
        while (done < 0) :
            j = np.random.randint(27)
            if bigram[i][j] > diff: 
                bigram[i][j] = round(bigram[i][j] - diff, 4)
                done = 1
    return bigram

def trigram_prob_fix(trigram):
    for i in range(27):
        for j in range(27):
            p_tot = round(sum(trigram[i][j]), 4)
            diff = p_tot - 1
            done = -1

            while (done < 0):
                k = np.random.randint(27)
                if trigram[i][j][k] > diff: 
                    trigram[i][j][k] = round(trigram[i][j][k] - diff, 4)
                    done = 1
    return trigram

def cdf_inv_calc(prob):
    inv = [0]*27
    inv[0] = prob[0]

    for i in range(1, 27): # given a set of probabilities, calculate the CDF inversion
        inv[i] = inv[i-1] + prob[i]
    return inv

def find_char(prob):
    cdf_inv = cdf_inv_calc(prob)
    p = np.random.random()

    for i in range(27): # find which char index corresponds to the random probability
        if p <= cdf_inv[i]: 
            return i

def add_char(char_index, s):
    return s + letters[char_index]

def q2_7_8(unigram, num):
    name = 'P3_Q' + str(num) + ".txt"
    q2_7_8 = ''
    for p in unigram: # store probabilities to a string
        q2_7_8 = q2_7_8 + str(p) + ', '
    q2_7_8 = q2_7_8[:-2] # remove last ', ' from string
    with open(name, 'w') as filehandle: # write unigram transition matrix to a text file
        filehandle.writelines("%s\n" % q2_7_8)
def q3_4(bigram, num):
    name = 'P3_Q' + str(num) + ".txt"
    with open(name, 'w') as filehandle: # write bigram transition matrix to a text file
        for i in range(27): # store probabilities to a string
            q3_4 = ''
            for p in bigram[i]:
                q3_4 = q3_4 + str(p) + ', '
            q3_4 = q3_4[:-2] # remove last ', ' from string
            filehandle.writelines("%s\n" % q3_4)
def q5(sentences):
    with open('P3_Q5.txt', 'w') as filehandle: # write sentences to a text file
        for s in sentences: 
            filehandle.writelines("%s\n" % s)
def q9(labels):
    q9 = ''
    for l in labels: # store probabilities to a string
        q9 = q9 + str(l) + ', '
    q9 = q9[:-2] # remove last ', ' from string
    with open('P3_Q9.txt', 'w') as filehandle: # write unigram transition matrix to a text file
        filehandle.writelines("%s\n" % q9)

############################################### CODE SETUP ###############################################
with open('Spirited Away Script.txt', encoding='utf-8') as filehandle:
    script = filehandle.read()

script = script.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
script = script.lower() # make everything lowercase
script = re.sub('[^a-z]+', ' ', script) # remove double spaces
script = ' '.join(script.split(' '))

with open('random_script.txt') as filehandle:
    r_script = filehandle.read()

indices = {
    ' ' : 0, 'a' : 1, 'b' : 2, 'c' : 3, 'd' : 4, 'e' : 5, 'f' : 6, 'g' : 7, 'h' : 8,  'i' : 9, 'j' : 10,
    'k' : 11, 'l' : 12, 'm' : 13, 'n' : 14, 'o' : 15, 'p' : 16, 'q' : 17, 'r' : 18, 's' : 19, 't' : 20,
    'u' : 21, 'v' : 22, 'w' : 23, 'x' : 24, 'y' : 25, 'z' : 26
}
letters = {v: k for k, v in indices.items()}

################################################# PART 1 #################################################
n = len(script) # total char counts
unigram = np.zeros(27) # unigram probabilities
bigram = np.zeros((27, 27)) # bigram probabilities
trigram = np.zeros((27, 27, 27)) # trigram probabilities

# store unigram, bigram, and trigram counts
c = [None, None, None]
for char in script: 
    c = [c[1], c[2], char] # reassign first, second, and third chars
    i = [indices[ch] if ch != None else None for ch in c] # get array indices of each char

    # update counts
    unigram[i[2]] = unigram[i[2]] + 1
    if i[1] != None: bigram[i[1]][i[2]] = bigram[i[1]][i[2]] + 1
    if i[0] != None and i[1] != None: trigram[i[0]][i[1]][i[2]] = trigram[i[0]][i[1]][i[2]] + 1

# loop script for 2nd to last character bigram and trigram probabilities
c = [c[1], c[2], script[0]] 
i = [indices[ch] for ch in c]
bigram[i[1]][i[2]] = bigram[i[1]][i[2]] + 1
trigram[i[0]][i[1]][i[2]] = trigram[i[0]][i[1]][i[2]] + 1

# loop script for last character bigram and trigram probabilities
c = [c[1], script[0], script[1]] 
i = [indices[ch] for ch in c]
bigram[i[1]][i[2]] = bigram[i[1]][i[2]] + 1
trigram[i[0]][i[1]][i[2]] = trigram[i[0]][i[1]][i[2]] + 1

bigram_no_smooth = np.zeros((27, 27)) # create array for bigram no smoothing transition matrix
# calculate unigram, bigram, and trigram transition probabilities
for i in range(27):
    for j in range(27):
        for k in range(27):
            trigram[i][j][k] = prob_calc(trigram[i][j][k], bigram[i][j], 27, True)

        bigram_no_smooth[i][j] = prob_calc(bigram[i][j], unigram[i], 0, False)
        bigram[i][j] = prob_calc(bigram[i][j], unigram[i], 27, True)
    
    unigram[i] = prob_calc(unigram[i], n, 0, False)

# fix probabilities (> or < 1.000) of bigram and trigram matrices
bigram = bigram_prob_fix(bigram)
bigram_no_smooth = bigram_prob_fix(bigram_no_smooth)
trigram = trigram_prob_fix(trigram)

# Question 2
q2_7_8(unigram, 2)

# Question 3
q3_4(bigram_no_smooth, 3)

# Question 4
q3_4(bigram, 4)

# generate 26 sentences based on transition matrices of the script
sentences = [""]*26
for i in range(26):
    # first character is a, b, c, etc.
    c1 = i + 1
    sentences[i] = sentences[i] + letters[c1]

    # find second character from bigram probability of c1
    c2 = find_char(bigram[c1])
    sentences[i] = sentences[i] + letters[c2]

    x = 2 # number of chars in the current sentence
    while (x < 1000):
        # find third character from trigram probability of c1, c2
        c3 = find_char(trigram[c1][c2])
        sentences[i] = sentences[i] + letters[c3]
        x = x + 1

        c1, c2 = c2, c3 # reassign c1, c2 for next iteration of chracter generation

# Question 5
q5(sentences)

################################################# PART 2 #################################################
r_n = len(r_script)
r_unigram = [0]*27

for char in r_script: # count instances of each char in the random script
    i = indices[char]
    r_unigram[i] = r_unigram[i] + 1

for i in range(27): # calculate unigram probabilities of random script
    r_unigram[i] = prob_calc(r_unigram[i], r_n, 0, False)

# Question 7
q2_7_8(r_unigram, 7)

# determine posterior probabilities of random script
p = 0.33
r_post = [0]*27
for i in range(27):
    r_post[i] = (p * r_unigram[i]) / (p * r_unigram[i] + ((1 - p) * unigram[i]))

# Question 8
q2_7_8(r_post, 8)

labels = [] # keep track of each sentence's classification
for s in sentences: # for each sentence, determine a label using a Naive Bayes classifier
    log_r = 0
    log_s = 0

    for i in range(len(s)): # iterate through each char in the sentence
        char = s[i]
        char_ind = indices[char]

        # update log likelihood of script and random script
        log_r = log_r + (i + 1) * math.log10(r_post[char_ind])
        log_s = log_s + (i + 1) * math.log10(1 - r_post[char_ind])

    # compare log likelihoods to determine label of sentence
    if log_s >= log_r: labels.append(0)
    else: labels.append(1)

# Question 9
q9(labels)