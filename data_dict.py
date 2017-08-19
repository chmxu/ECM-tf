import copy
import json
import pickle
def read_train_set():
    file_name = "train_data.json"
    f = open(file_name, encoding="utf-8")
    train_data = json.load(f)
    f.close()
    return train_data


def update_word_dict(word_dict, word, emotion):
    if word in word_dict.keys():
        word_dict[word][emotion] += 1
    else:
        word_dict[word] = [0,0,0,0,0,0]
        word_dict[word][emotion] += 1
    return word_dict

def read_word(data):
    word_dict = dict()
    for q_a in data:
        for words, emotion in q_a:
            for word in words:
                word_dict = update_word_dict(word_dict, word.rstrip(), emotion)
    return word_dict

def data_str2list(data):
    for question, answer in data:
        question[0] = question[0].split(" ")
        answer[0] = answer[0].split(" ")

    return data
def questions_answers(data):
    questions, answers ,questions_emotion,answers_emotion= [], [], [],[]
    for q_a in data:
        q_a[0][0][-1] = q_a[0][0][-1].rstrip()
        q_a[1][0][-1] = q_a[1][0][-1].rstrip()
        q_a[0][0].append('<END>')
        q_a[1][0].append('<END>')
        questions.append(q_a[0][0])
        questions_emotion.append(q_a[0][1])
        answers.append(q_a[1][0])
        answers_emotion.append(q_a[1][1])

    assert len(questions) == len(answers)
    return questions, answers,questions_emotion,answers_emotion


def trans(x_list,index_dict):
    idx_list = []
    for i in range(len(x_list)):
        idx_list.append(index_dict[x_list[i]])
    return idx_list

def trans_questions_answers(questions, answers, index_dict):
    trans_questions, trans_answers = [], []
    for question in questions:
        trans_questions.append(trans(question, index_dict))
    for answer in answers:
        trans_answers.append(trans(answer, index_dict))
    assert len(trans_questions) == len(trans_answers)
    return trans_questions, trans_answers

def data_dict(questions, answers,trans_questions, trans_answers,questions_emotion,answers_emotion):
    datadict = dict()
    datadict['questions'] = questions
    datadict['answers'] = answers
    datadict['trans_questions'] = trans_questions
    datadict['trans_answers'] = trans_answers
    datadict['questions_emotion'] = questions_emotion
    datadict['answers_emotion'] = answers_emotion
    return datadict

data = data_str2list(read_train_set()) #读入数据，去掉空格
print(data[0])


word_dict = read_word(data)
# print(word_dict)
#选出top1万出现的词
for i in word_dict.keys():
    word_dict[i] = sum(word_dict[i])


new_word_list= sorted(word_dict.items(), key=lambda d:d[1], reverse = True)
new_word_dict = dict()
for i in range(len(new_word_list)):
    new_word_dict[new_word_list[i][0]] = i

new_word_dict['<START>'] = len(new_word_dict)
new_word_dict['<END>'] = len(new_word_dict)
new_word_dict['<NULL>'] = len(new_word_dict)

f = open("word2idx_origin.pkl", 'wb')
pickle.dump(new_word_dict, f)
f.close()
index_list = list(word_dict.keys())
# index_dict
index_dict = dict()
for i in range(len(index_list)):
    index_dict[index_list[i]] = i

questions, answers ,questions_emotion,answers_emotion = questions_answers(data)
#找到筛选的index
del_list =[]

for i in range(len(questions)):
    for j in questions[i]:
        if j not in new_word_dict.keys():
            del_list.append(i)
            break
for i in range(len(answers)):
    for j in answers[i]:
        if j not in new_word_dict.keys():
            del_list.append(i)
            break

del_list = set(del_list)
data_dict_ = dict()
for i in range(len(questions)):
    data_dict_[i] = {"question": questions[i], "answer": answers[i], "question_emotion": questions_emotion[i],
                   "answer_emotion": answers_emotion[i]}
for elem in del_list:
    data_dict_.pop(elem)

questions = []
answers = []
questions_emotion = []
answers_emotion = []
for data in data_dict_.values():
    questions.append(data["question"])
    answers.append(data["answer"])
    questions_emotion.append(data["question_emotion"])
    answers_emotion.append(data["answer_emotion"])


trans_questions, trans_answers = trans_questions_answers(questions, answers, new_word_dict)


datadict = data_dict(questions, answers,trans_questions, trans_answers,questions_emotion,answers_emotion)
f = open("data_origin.pkl", 'wb')
pickle.dump(datadict, f)
f.close()
# print(datadict)
