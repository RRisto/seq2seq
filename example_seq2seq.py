from seq2seq.data.data_manager import Seq2SeqDataManager
from seq2seq.model.seq2seq_learner import Seq2seqLearner

DEVICE = 'cpu'
MIN_LENGTH = 3
MAX_LENGTH = 10
MIN_COUNT = 3

## Get data
data_manager = Seq2SeqDataManager.create_from_txt('data/eng-fra_sub.txt','en', 'fr', min_freq=MIN_COUNT, min_ntoks=MIN_LENGTH,
                                                  max_ntoks=MAX_LENGTH, switch_pair=True)

#data_manager.save('example_data_manager.pth')
#data_manager=Seq2SeqDataManager.load('example_data_manager.pth')
hidden_size=50
learner=Seq2seqLearner(data_manager,hidden_size)
learner.fit(20, show_attention_every=5)

original_xtext = 'Je suis sûr.'
original_ytext = 'I am sure.'
predicted_text = learner.predict(original_xtext, device=DEVICE)
print(f'original text: {original_xtext}')
print(f'original answer: {original_ytext}')
print(f'predicted text: {predicted_text}')