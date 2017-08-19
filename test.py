import pickle
import argparse
import sys
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="the model chosen", default="internal", type=str)
args = parser.parse_args()
model_chosen = args.model
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
f = open('data/data_test.pkl', 'rb')
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
for question in data['trans_questions']:
	question.extend([word2idx['<NULL>']] * (max_length_questions-len(question)))
for answer in data['trans_answers']:
	answer.extend([word2idx['<NULL>']] * (max_length-len(answer)))
batch_size = 128
n_iters_per_epoch = int(np.ceil(float(len(data['questions'])) / batch_size))
model = ECM_Model(max_length_questions, max_length, emotion_num=6, word_to_idx=word2idx, embedding_matrix=None, learning_rate=0.5)

solver = ECMSolver(model, data, word2idx=word2idx, val_data=None, n_epochs=1, batch_size=batch_size,
                                    print_every=int(n_iters_per_epoch/2), save_every=100,
                                    pretrained_model=None, model_path='model/lstm/'+model_chosen,
                                    test_model='model/'+model_chosen+'/model',
                                    log_path='log/')
new_data = solver.test()
f = open("test_result_"+model_chosen+".pkl", 'wb')
pickle.dump(new_data, f)
f.close()
print(new_data["questions"][0])
print(new_data["answers"][0])
print(new_data["ground_truth"][0])