import pickle
import argparse
import sys
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="the model chosen", default="internal", type=str)
parser.add_argument("-b", "--batch_size", help="batch size", default=128, type=int)
parser.add_argument("-l", "--learning_rate", help="learning rate", default=0.1, type=float)
parser.add_argument("-p", "--pretrained_model", help="pretrained model", default=None, type=str)
parser.add_argument("-e", "--epoch", help="number of epoches", default=200, type=int)
args = parser.parse_args()
model_chosen = args.model
batch_size = args.batch_size
learning_rate = args.learning_rate
pretrained_model = args.pretrained_model
epoch = args.epoch
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
f = open('data_train.pkl', 'rb')
data = pickle.load(f)
f.close()
f = open('word2idx.pkl', 'rb')
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
n_iters_per_epoch = int(np.ceil(float(len(data['questions'])) / batch_size))
model = ECM_Model(max_length_questions, max_length, emotion_num=6, word_to_idx=word2idx, embedding_matrix=None, learning_rate=learning_rate)
solver = ECMSolver(model, data, word2idx=word2idx, val_data=None, n_epochs=epoch, batch_size=batch_size,
                                    print_every=int(n_iters_per_epoch/15), save_every=5,
                                    pretrained_model=pretrained_model, model_path='model/'+model_chosen,
                                    test_model=None,
                                    log_path='log/')
solver.train()