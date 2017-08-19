import pickle
from internal_mem.model_revise import ECM_Model
from internal_mem.solver_revise import ECMSolver

f = open('data_mini.pkl', 'rb')
data = pickle.load(f)
f.close()
f = open('word2idx_mini.pkl', 'rb')
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

model = ECM_Model(max_length_questions, max_length, emotion_num=6, word_to_idx=word2idx, embedding_matrix=None, learning_rate=0.02)
solver = ECMSolver(model, data, word2idx=word2idx, val_data=None, n_epochs=200, batch_size=128, update_rule='adam',
                                    print_every=1, save_every=1,
                                    pretrained_model=None, model_path='model/lstm/',
                                    test_model='model/lstm/model-10',
                                    log_path='log/')
solver.train()