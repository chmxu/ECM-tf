import pickle
import argparse
import sys
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="the model chosen", default="internal", type=str)
args = parser.parse_args()
model_chosen = args.model
if model_chosen == "internal":
	from internal_mem import model_revise
	from internal_mem import solver_revise
	ECM_Model = model_revise.ECM_Model
	ECMSolver = solver_revise.ECMSolver
else:
	from emo_embedding import model_revise
	from emo_embedding import solver_revise
	ECM_Model = model_revise.ECM_Model
	ECMSolver = solver_revise.ECMSolver
f = open('data_medium.pkl', 'rb')
data = pickle.load(f)
f.close()
f = open('word2idx_medium.pkl', 'rb')
word2idx = pickle.load(f)
max_value = max(word2idx.values())
f.close()
max_length_questions = 0
max_length = 0
for question in data['questions']:
	max_length_questions = max(max_length_questions, len(question))
for answer in data['answers']:
	max_length = max(max_length, len(answer))
for question in data['trans_questions']:
	question.extend([word2idx['<NULL>']] * (max_length_questions-len(question)))
for answer in data['trans_answers']:
	answer.extend([word2idx['<NULL>']] * (max_length-len(answer)))
batch_size = 128
question = ["知道", "真相", "的", "我", "眼泪", "掉", "下来"]
question.extend(['<NULL>'] * (max_length_questions-len(question)))
trans_question = []
for word in question:
	trans_question.append(word2idx[word])
new_data = {'questions': [question]*6, "trans_questions": [trans_question]*6, "questions_emotion": [0,1,2,3,4,5]}
n_iters_per_epoch = int(np.ceil(float(len(data['questions'])) / batch_size))
model = ECM_Model(max_length_questions, max_length, emotion_num=6, word_to_idx=word2idx, embedding_matrix=None, learning_rate=0.5)

solver = ECMSolver(model, new_data, word2idx=word2idx, val_data=None, n_epochs=2000, batch_size=batch_size, update_rule='adam',
                                    print_every=int(n_iters_per_epoch/2), save_every=100,
                                    pretrained_model=None, model_path='model/lstm/'+model_chosen,
                                    test_model='model/lstm/internal/model-310',
                                    log_path='log/')
response = solver.apply()
print(response)