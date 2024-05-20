FROM python
COPY . /MNIST_API/
RUN pip install --no-cache-dir torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r /MNIST_API/requirements.txt --no-cache-dir
EXPOSE 8000
WORKDIR /MNIST_API/
CMD ["python","app.py"]