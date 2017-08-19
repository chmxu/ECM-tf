import pickle
import argparse
import numpy as np
import sys
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="the model chosen", default="internal", type=str)
parser.add_argument("-p", "--model_path", help="the path of the model", default=None, type=str)
args = parser.parse_args()
model_chosen = args.model
model_path = args.model_path
if model_path is None:
	model_path = 'model/'+model_chosen+'/model'
if model_chosen == "internal":
	from internal_mem import model_revise, solver_revise
	ECM_Model = model_revise.ECM_Model
	ECMSolver = solver_revise.ECMSolver
elif model_chosen == "ECM":
	from ECM import model_revise, solver_revise
	ECM_Model = model_revise.ECM_Model
	ECMSolver = solver_revise.ECMSolver
else:
	from emo_embedding import model_revise, solver_revise
	ECM_Model = model_revise.ECM_Model
	ECMSolver = solver_revise.ECMSolver
f = open('data/data_train.pkl', 'rb')
data = pickle.load(f)
f.close()
f = open('data/word2idx.pkl', 'rb')
word2idx = pickle.load(f)
max_value = max(word2idx.values())
f.close()
max_length_questions = 0
max_length = 0
for question in data['questions']:
	max_length_questions = max(max_length_questions, len(question))
for answer in data['answers']:
	max_length = max(max_length, len(answer))
batch_size = 128
model = ECM_Model(max_length_questions, max_length, emotion_num=6, word_to_idx=word2idx, embedding_matrix=None,
                  learning_rate=0.5)

solver = ECMSolver(model, data, word2idx=word2idx, val_data=None, n_epochs=1, batch_size=batch_size,
                   print_every=1, save_every=100,
                   pretrained_model=None, model_path='model/'+model_chosen+'/0.05',
                   test_model=model_path,
                   log_path='log/')
time = 0
while True:
	print("post")
	sys.stdout.flush()
	question = input()
	question = question.split(" ")
	question.append('<END>')
	question.extend(['<NULL>'] * (max_length_questions - len(question)))
	trans_question = []
	flag = 0
	for word in question:
		try:
			trans_question.append(word2idx[word])
		except:
			flag = 1
			print("The word %s is not in the dictionary! Try other post." % word)
			break
	if flag == 1:
		continue
	print("emotion category, 1 for like, 2 for sadness, 3 for disgust, 4 for anger, 5 for happiness and 0 for other")
	sys.stdout.flush()
	flag = 0
	try:
		emotion = int(input())
	except:
		print("emotion should be a number fro 0 to 5!")
		continue
	if emotion > 5 or emotion < 0:
		print("emotion should be a number fro 0 to 5!")
		continue
	new_data = {'questions': [question] * 2, "trans_questions": [trans_question] * 2,
	            "questions_emotion": [emotion]*2}

	n_iters_per_epoch = int(np.ceil(float(len(data['questions'])) / batch_size))
	solver.data = new_data
	response = solver.apply(time)
	if len(response[0]) > len(response[1]):
		print(response[0])
	else:
		print(response[1])
	time += 1
