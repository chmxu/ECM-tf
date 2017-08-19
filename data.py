import pickle
f = open('data.pkl', 'rb')
data = pickle.load(f)
questions = data['questions']
answers = data['answers']
word_dict = dict()
for question in questions:
	for word in question:
		if not(word in word_dict.keys()):
			word_dict[word] = len(word_dict)
word_dict['<NULL>'] = len(word_dict)
output = open('word2idx.pkl', 'wb')
pickle.dump(word_dict, output)
output.close()
trans_questions = []
trans_answers = []
for question in questions:
	trans_questions.append([word_dict[word] for word in question])
for answer in answers:
	trans_list = []
	for word in answer:
		if word in word_dict.keys():
			trans_list.append(word_dict[word])
		else:
			trans_list.append(word_dict['<NULL>'])
	trans_answers.append(trans_list)
new_dict = {'questions': questions, 'answers': answers, 'trans_questions': trans_questions, 'trans_answers': trans_answers,
            'questions_emotion': data['questions_emotion']}
output = open('data.pkl', 'wb')
pickle.dump(new_dict, output)
output.close()