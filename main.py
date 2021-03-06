import streamlit as st
from random import randint
from pickle import load
from tensorflow.keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import zipfile, io
import warnings
warnings.filterwarnings("ignore")

def load_lines():
    with zipfile.ZipFile("./republic_sequences.zip") as zf:
        with io.TextIOWrapper(zf.open("republic_sequences.txt"), encoding="utf-8") as f:
            data = f.read()
            lines = data.split('\n')
    return lines

lines = load_lines()

def generate_sequ():
    seed_text = lines[randint(0,len(lines))]
    return seed_text

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text

    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]

        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating= 'pre' )

        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)

        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text += ' ' + out_word
        result.append(out_word)
    return ' ' .join(result)

st.title("Language Modelling API")

# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

seed_text = generate_sequ()

nwords = st.slider("Adjust number of words generated in a sequence", min_value=1, max_value=50, step=1, value=10)

# generate new text
generated = generate_seq(model, tokenizer, 50, seed_text, nwords)

stout = st.empty()

col1, col2 = st.beta_columns([2,8])
with col1:
    btn_text = st.button("Generate Seq")
    if btn_text:
        with col2:
            stout.markdown(seed_text + "**_"+ " " +generated+"_**")

