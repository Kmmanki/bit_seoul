from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 맛있는 밥을 진짜 먹었다.'
token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# text = '나는 울트라 맛있는 밥을 먹었다.'
# {'나는': 1, '울트라': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}
# text = '나는 진짜 맛있는 밥을 진짜 먹었다.'
# {'진짜': 1, '나는': 2, '맛있는': 3, '밥을': 4, '먹었다': 5}

x = token.texts_to_sequences([text])
print(x) #[[2, 1, 3, 4, 1, 5]]

from tensorflow.keras.utils import to_categorical

word_size = len(token.word_index)
x = to_categorical(x, num_classes=word_size+1)
print(x)