import streamlit as st
from model import model_predict

def main():
    st.title("Audio Sentiment Analysis App")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])
    output_file = None

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Save the uploaded file to disk
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        output_file = "uploaded_audio.wav"
        st.success("File uploaded and saved successfully!")

    if st.button("Predict Emotion"):
        if output_file == None:
            st.write("Please upload an audio file first.")
        else:
            emotion = model_predict(output_file)
            st.write("Predicted emotion:", emotion)

if __name__ == "__main__":
    main()