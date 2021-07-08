import streamlit as st 
from PIL import Image
from nmt import*





image = Image.open('images/image.jpeg')

st.image(image, caption='Natural Machine Translation', width=50,  use_column_width=True) 

input_text = st.text_area('', height=100)



st.write(input_text)


 
states_values = enc_model.predict( str_to_tokens( input_text) )
empty_target_seq = np.zeros( ( 1 , 1 ) )
empty_target_seq[0, 0] = french_word_index['start']
stop_condition = False
decoded_translation = ''
while not stop_condition :
    dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
    sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
    sampled_word = None
    for word , index in french_word_index.items() :
        if sampled_word_index == index :
            decoded_translation += ' {}'.format( word )
            sampled_word = word
        
    if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length_sequences:
        stop_condition = True
            
    empty_target_seq = np.zeros( ( 1 , 1 ) )  
    empty_target_seq[ 0 , 0 ] = sampled_word_index
    states_values = [ h , c ] 

if st.button('Translate'):
    st.success(decoded_translation)





hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

 