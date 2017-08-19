import random
from sklearn.cross_validation import train_test_split
import pickle
f = open("data.pkl", 'rb')
data = pickle.load(f)
length = len(data['questions'])
print(length)
train, test, _, _ = train_test_split(range(length),
                                  range(length),
                                  test_size = 0.1,
                                  random_state = 0)
train_data = {"questions":[], "answers":[], "trans_questions":[], "trans_answers":[], "questions_emotion":[]}
test_data = {"questions":[], "answers":[], "trans_questions":[], "trans_answers":[], "questions_emotion":[]}
for item in train:
	train_data["questions"].append(data['questions'][item])
	train_data["answers"].append(data['answers'][item])
	train_data["trans_questions"].append(data['trans_questions'][item])
	train_data["trans_answers"].append(data['trans_answers'][item])
	train_data["questions_emotion"].append(data['questions_emotion'][item])
for item in test:
	test_data["questions"].append(data['questions'][item])
	test_data["answers"].append(data['answers'][item])
	test_data["trans_questions"].append(data['trans_questions'][item])
	test_data["trans_answers"].append(data['trans_answers'][item])
	test_data["questions_emotion"].append(data['questions_emotion'][item])
w = open("data_train.pkl", "wb")
pickle.dump(train_data, w)
print(len(train_data['questions']))
w.close()
w = open("data_test.pkl", 'wb')
pickle.dump(test_data, w)
print(len(test_data['questions']))
w.close()