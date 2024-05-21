FROM pytorch/pytorch

WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir gradio transformers pillow>=9.4.0 
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
